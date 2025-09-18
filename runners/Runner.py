import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from models.GloCTM import GloCTM


class Runner:
    def __init__(self, args):
        self.args = args
        self.model = GloCTM(args)

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{args.device}" if args.device is not None else "cuda:0")
            self.model = self.model.to(self.device)

    def make_optimizer(self):
        args_dict = {
            'params': self.model.parameters(),
            'lr': self.args.learning_rate
        }

        optimizer = torch.optim.Adam(**args_dict)
        return optimizer

    def make_lr_scheduler(self, optimizer):
        if self.args.lr_scheduler == 'StepLR':
            lr_scheduler = StepLR(optimizer, step_size=self.args.lr_step_size, gamma=self.args.lr_gamma, verbose=False)
        else:
            raise NotImplementedError(self.args.lr_scheduler)

        return lr_scheduler

    def train(self, data_loader):
        data_size = len(data_loader.dataset)
        num_batch = len(data_loader)
        optimizer = self.make_optimizer()

        if hasattr(self.args, 'lr_scheduler') and self.args.lr_scheduler:
            lr_scheduler = self.make_lr_scheduler(optimizer)

        for epoch in range(1, self.args.epochs + 1):
            sum_loss = 0.
            loss_rst_dict = defaultdict(float)
            self.model.train()

            for batch_data in data_loader:
                # Local Bag-of-Words
                batch_bow_en = batch_data['bow_en']
                batch_bow_cn = batch_data['bow_cn']
                batch_global_bow_en = batch_data['global_bow_en']
                batch_global_bow_cn = batch_data['global_bow_cn']
                
                # Move tensors to device if needed
                if torch.cuda.is_available() and hasattr(self, 'device'):
                    batch_bow_en = batch_bow_en.to(self.device)
                    batch_bow_cn = batch_bow_cn.to(self.device)
                    batch_global_bow_en = batch_global_bow_en.to(self.device)
                    batch_global_bow_cn = batch_global_bow_cn.to(self.device)
                
                # Get document embeddings for CKA loss
                document_info = None
                if 'doc_embedding_en' in batch_data and 'doc_embedding_cn' in batch_data:
                    doc_emb_en = batch_data['doc_embedding_en']
                    doc_emb_cn = batch_data['doc_embedding_cn']
                    
                    # Move document embeddings to device
                    if torch.cuda.is_available() and hasattr(self, 'device'):
                        doc_emb_en = doc_emb_en.to(self.device)
                        doc_emb_cn = doc_emb_cn.to(self.device)
                    
                    document_info = {
                        'doc_embedding_en': doc_emb_en,
                        'doc_embedding_cn': doc_emb_cn
                    }

                # Prepare parameters for the model's forward pass
                params_list = [
                    batch_bow_en, batch_bow_cn, 
                    batch_global_bow_en, batch_global_bow_cn
                ]
                
                # Add optional parameters
                kwargs = {}
                if document_info is not None:
                    kwargs['document_info'] = document_info

                rst_dict = self.model(*params_list, **kwargs)

                batch_loss = rst_dict['loss']

                # Accumulate losses for logging
                for key in rst_dict:
                    if 'loss' in key:
                        loss_rst_dict[key] += rst_dict[key].item()

                # Backward pass and optimization
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                sum_loss += batch_loss.item() * len(batch_bow_en)

            if hasattr(self.args, 'lr_scheduler') and self.args.lr_scheduler:
                lr_scheduler.step()

            # --- Logging for the epoch ---
            output_log = f'Epoch: {epoch:03d}'
            for key, value in loss_rst_dict.items():
                output_log += f' {key}: {value / num_batch:.4f}'
            print(output_log)

        # After training, get the learned topic-word distributions (beta)
        beta_en, beta_cn = self.model.get_beta()
        beta_en = beta_en.detach().cpu().numpy()
        beta_cn = beta_cn.detach().cpu().numpy()
        return beta_en, beta_cn

    def get_theta(self, bow, lang):
        theta_list = list()
        data_size = bow.shape[0]
        all_idx = torch.split(torch.arange(data_size,), self.args.batch_size)
        
        # Move bow to device if needed
        if torch.cuda.is_available() and hasattr(self, 'device'):
            bow = bow.to(self.device)
            
        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_bow = bow[idx]
                result = self.model.get_theta(batch_bow, lang)
                # Handle both tuple and tensor returns
                if isinstance(result, tuple):
                    theta = result[0]  # Get first element (theta)
                else:
                    theta = result
                theta_list.extend(theta.detach().cpu().numpy().tolist())
        return np.asarray(theta_list)

    def test(self, dataset):
        theta_en = self.get_theta(dataset.bow_en, lang='en')
        theta_cn = self.get_theta(dataset.bow_cn, lang='cn')
        return theta_en, theta_cn