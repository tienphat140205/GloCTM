import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.Encoder import Encoder


class CKALoss(nn.Module):
    """
    Loss with knowledge distillation.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, SH, TH): 
        dT = TH.size(-1)
        dS = SH.size(-1)
        SH = SH.view(-1, dS).to(SH.device, torch.float64)
        TH = TH.view(-1, dT).to(SH.device, torch.float64)
        
        slen = SH.size(0)
        # Dropout on Hidden State Matching
        SH = SH - SH.mean(0, keepdim=True)
        TH = TH - TH.mean(0, keepdim=True)
                
        num = torch.norm(SH.t().matmul(TH), 'fro')
        den1 = torch.norm(SH.t().matmul(SH), 'fro') + self.eps
        den2 = torch.norm(TH.t().matmul(TH), 'fro') + self.eps
        
        return 1 - num/torch.sqrt(den1*den2)


class GloCTM(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.vocab_en = args.vocab_en
        self.vocab_cn = args.vocab_cn
        
        # Initialize beta matrices with Xavier uniform
        beta_en = torch.empty(args.num_topic, len(self.vocab_en))
        beta_cn = torch.empty(args.num_topic, len(self.vocab_cn))
        nn.init.xavier_uniform_(beta_en)
        nn.init.xavier_uniform_(beta_cn)
        self.beta_en = nn.Parameter(beta_en)
        self.beta_cn = nn.Parameter(beta_cn)
        
        self.vocab_size_en = len(self.vocab_en)
        self.vocab_size_cn = len(self.vocab_cn)
        self.num_topic = args.num_topic
        self.weight_kl_loss = args.weight_kl_loss
        
        # CKA loss weight and module
        self.weight_cka_loss = getattr(args, 'weight_cka_loss', 300)  # Default weight
        self.cka_loss = CKALoss(eps=1e-8)
        
        # Fixed priors for GLOBAL pathway
        self.a = 1 * np.ones((1, int(args.num_topic))).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T), requires_grad=False)
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / args.num_topic))).T + (1.0 / (args.num_topic * args.num_topic)) * np.sum(1.0 / self.a, 1)).T), requires_grad=False)

        
        # Decoder BatchNorm layers (frozen weights)
        self.decoder_bn_en = nn.BatchNorm1d(self.vocab_size_en, affine=True)
        self.decoder_bn_en.weight.requires_grad = False
        self.decoder_bn_cn = nn.BatchNorm1d(self.vocab_size_cn, affine=True)
        self.decoder_bn_cn.weight.requires_grad = False
        self.decoder_bn_global = nn.BatchNorm1d(self.vocab_size_en + self.vocab_size_cn, affine=True)
        self.decoder_bn_global.weight.requires_grad = False

        # Encoders for GLOBAL and LOCAL pathways
        self.encoder_en = Encoder(self.vocab_size_en, self.num_topic, args.en1_units, args.dropout)
        self.encoder_cn = Encoder(self.vocab_size_cn, self.num_topic, args.en1_units, args.dropout)
        self.encoder_global = Encoder(self.vocab_size_en + self.vocab_size_cn, self.num_topic, args.en_units, args.dropout)

    def get_beta_global(self):
        """Returns concatenated global beta matrix from beta_en and beta_cn."""
        return torch.cat((self.beta_en, self.beta_cn), dim=1)

    def get_theta(self, x, lang):
        """Encodes input x to get topic distribution theta."""
        # Direct encoder access without getattr
        if lang == 'en':
            encoder = self.encoder_en
        elif lang == 'cn':
            encoder = self.encoder_cn
        else:  # global
            encoder = self.encoder_global
            
        theta, mu, logvar = encoder(x)
        return (theta, mu, logvar) if self.training else mu

    def decode(self, theta, beta, lang):
        """Decodes topic distribution theta to bag-of-words space."""
        # Direct access to batch norm layers
        if lang == 'en':
            bn_layer = self.decoder_bn_en
        elif lang == 'cn':
            bn_layer = self.decoder_bn_cn
        else:  # global
            bn_layer = self.decoder_bn_global
            
        return F.softmax(bn_layer(torch.matmul(theta, beta)), dim=1)

    def loss_vae_global(self, recon_x, x, mu, logvar):
        """VAE loss for GLOBAL pathway with fixed prior."""
        # Reconstruction loss
        recon_loss = -(x * torch.clamp(recon_x, min=1e-10).log()).sum(1)
        
        # KL divergence against fixed prior
        var = logvar.exp()
        diff = mu - self.mu2
        kl_loss = 0.5 * ((var / self.var2 + diff * diff / self.var2 + 
                         self.var2.log() - logvar).sum(1) - self.num_topic)
        
        total_loss = (recon_loss + kl_loss).mean()
        return total_loss, recon_loss.mean(), kl_loss.mean()
    
    def get_beta(self):
        beta_en = self.beta_en
        beta_cn = self.beta_cn
        return beta_en, beta_cn

    def loss_vae_local(self, recon_x, x, mu_local, logvar_local):
        """VAE loss for LOCAL pathway with same fixed prior as global."""
        # Reconstruction loss
        recon_loss = -(x * torch.clamp(recon_x, min=1e-10).log()).sum(1)
        
        # KL divergence against fixed prior
        var = logvar_local.exp()
        diff = mu_local - self.mu2
        kl_loss = 0.5 * ((var / self.var2 + diff * diff / self.var2 + 
                         self.var2.log() - logvar_local).sum(1) - self.num_topic)
        
        total_loss = (recon_loss + kl_loss).mean()
        return total_loss, recon_loss.mean(), kl_loss.mean()

    def loss_kl_global_local(self, mu_local, logvar_local, mu_global, logvar_global):
        """KL divergence between global and local theta distributions."""
        var_local = logvar_local.exp()
        var_global = logvar_global.exp()
        diff = mu_local - mu_global
        kl_loss = 0.5 * (var_local / var_global + diff * diff / var_global - 1 + 
                        logvar_global - logvar_local).sum(1)
        return kl_loss.mean()

    def forward(self, x_en, x_cn, x_g_en, x_g_cn, document_info=None, global_indices_en=None, global_indices_cn=None):
        """
        Main forward pass with individual parameters.
        Args:
            x_en: English BOW
            x_cn: Chinese BOW
            x_g_en: Global English BOW
            x_g_cn: Global Chinese BOW
            document_info: Dict containing document embeddings {'doc_embedding_en': ..., 'doc_embedding_cn': ...}
            global_indices_en: Global indices for English documents
            global_indices_cn: Global indices for Chinese documents
        """
        # --- GLOBAL PATHWAY ---
        theta_g_en, mu_g_en, logvar_g_en = self.get_theta(x_g_en, 'global')
        theta_g_cn, mu_g_cn, logvar_g_cn = self.get_theta(x_g_cn, 'global')
        
        # Use single beta_g for both languages
        beta_g = self.get_beta_global()
        recon_x_g_en = self.decode(theta_g_en, beta_g, 'global')
        recon_x_g_cn = self.decode(theta_g_cn, beta_g, 'global')
        
        # Lấy phần tử đầu tiên của tuple (total_loss)
        loss_g_en, _, _ = self.loss_vae_global(recon_x_g_en, x_g_en, mu_g_en, logvar_g_en)
        loss_g_cn, _, _ = self.loss_vae_global(recon_x_g_cn, x_g_cn, mu_g_cn, logvar_g_cn)
        loss_g = loss_g_en + loss_g_cn
        
        # --- LOCAL PATHWAY ---
        theta_en, mu_en, logvar_en = self.get_theta(x_en, 'en')
        theta_cn, mu_cn, logvar_cn = self.get_theta(x_cn, 'cn')
        
        # Generate reconstructions
        recon_x_en = self.decode(theta_en, self.beta_en, 'en')
        recon_x_cn = self.decode(theta_cn, self.beta_cn, 'cn')
        
        # Calculate VAE losses with same prior as global
        loss_l_en, _, _ = self.loss_vae_local(recon_x_en, x_en, mu_en, logvar_en)
        loss_l_cn, _, _ = self.loss_vae_local(recon_x_cn, x_cn, mu_cn, logvar_cn)
        
        # Calculate KL loss between global and local theta distributions
        kl_loss_en = self.loss_kl_global_local(mu_en, logvar_en, mu_g_en, logvar_g_en)
        kl_loss_cn = self.loss_kl_global_local(mu_cn, logvar_cn, mu_g_cn, logvar_g_cn)
        kl_loss = (kl_loss_en + kl_loss_cn) * self.weight_kl_loss  # Use weight from args
        
        # --- CKA LOSS between LOCAL THETA and DOCUMENT EMBEDDINGS ---
        cka_loss_total = torch.tensor(0.0, device=x_en.device)
        
        if document_info is not None:
            # CKA loss for English documents - using LOCAL theta instead of global
            if 'doc_embedding_en' in document_info:
                doc_emb_en = document_info['doc_embedding_en']
                if doc_emb_en.size(0) == theta_en.size(0):  # Ensure batch sizes match
                    cka_loss_en = self.cka_loss(theta_en, doc_emb_en)
                    cka_loss_total += cka_loss_en
            
            # CKA loss for Chinese documents - using LOCAL theta instead of global
            if 'doc_embedding_cn' in document_info:
                doc_emb_cn = document_info['doc_embedding_cn']
                if doc_emb_cn.size(0) == theta_cn.size(0):  # Ensure batch sizes match
                    cka_loss_cn = self.cka_loss(theta_cn, doc_emb_cn)
                    cka_loss_total += cka_loss_cn
        
        # Apply CKA loss weight
        cka_loss_weighted = cka_loss_total * self.weight_cka_loss
        
        # --- TOTAL LOSS ---
        total_loss = loss_g + loss_l_en + loss_l_cn + kl_loss + cka_loss_weighted
        
        return {
            'loss': total_loss,
            'loss_global': loss_g,
            'loss_local_en': loss_l_en, 
            'loss_local_cn': loss_l_cn,
            'loss_kl_global_local': kl_loss,
            'loss_cka': cka_loss_weighted
        }