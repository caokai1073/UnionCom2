import time
import torch
import random
import numpy as np
import scipy.sparse as sp 
from sklearn.neighbors import NearestNeighbors
from torch.nn.functional import softmax, cosine_similarity, relu
from sklearn.metrics.pairwise import pairwise_distances, rbf_kernel
from sklearn.decomposition import PCA
import torch.multiprocessing as mp
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from itertools import cycle
import faiss
import math
from torch.utils.data import Dataset, DataLoader

class SigmaDataset(Dataset):
    def __init__(self, sigma, sigma_, neighbor_indices):
        self.sigma = sigma
        self.sigma_ = sigma_
        self.neighbor_indices = neighbor_indices

    def __len__(self):
        return len(self.sigma)

    def __getitem__(self, idx):
        return self.sigma[idx], self.sigma_[idx], self.neighbor_indices[idx]

class unioncom(object):
    def __init__(
        self, 
        manual_seed=1234, 
        kmin=20, 
        kmax=40, 
        distance_mode ='geodesic', 
    ):
        
        self.manual_seed = manual_seed
        self.kmax = kmax
        self.kmin = kmin
        self.distance_mode = distance_mode
    
    def init_random_seed(self, manual_seed):
        seed = None
        if manual_seed is None:
            seed = random.randint(1,10000)
        else:
            seed = manual_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
    def geodesic_distances(self, X, kmin=5, kmax=20, data_mode='query'):
        nbrs = NearestNeighbors(n_neighbors=kmin, metric='euclidean', n_jobs=-1).fit(X)
        knn = nbrs.kneighbors_graph(X, mode='distance')
        # connected_components = sp.csgraph.connected_components(knn, directed=False)[0]
        # while connected_components != 1:
        #     if kmin > np.max((kmax, 0.01*len(X))):    
        #         break
        #     kmin += 1
        #     nbrs = NearestNeighbors(n_neighbors=kmin, metric='euclidean', n_jobs=-1).fit(X)
        #     knn = nbrs.kneighbors_graph(X, mode='distance')
        #     connected_components = sp.csgraph.connected_components(knn, directed=False)[0]
        # print('connected components of '+data_mode+':', connected_components)
        # print('k of '+data_mode+':', kmin)
        dist = sp.csgraph.floyd_warshall(knn, directed=False)

        dist_max = np.nanmax(dist[dist != np.inf])
        dist[dist > dist_max] = 2*dist_max

        return dist

    def cal_data_with_neighbors(self, coord, k):
        knn = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean', n_jobs=-1).fit(coord)
        _, indices = knn.kneighbors(coord)
        return torch.from_numpy(indices)

    def cal_COVET(self, X, indices):
        E = X[indices]
        data_mean = torch.mean(E, dim=1)
        data_mean = data_mean.unsqueeze(1)
        diff = E - data_mean
        sigma = torch.einsum('ijk,ijl->ikl', diff, diff) / E.shape[1]
        return sigma
    
    def cal_self_similarity(self, centers1, centers2, device):
        if self.distance_mode == 'geodesic':
            geo1 = self.geodesic_distances(centers1, self.kmin, self.kmax, 'query')
            geo2 = self.geodesic_distances(centers2, self.kmin, self.kmax, 'reference')
        elif self.distance_mode == 'euclidean':
            geo1 = pairwise_distances(centers1, metric='euclidean')
            geo2 = pairwise_distances(centers2, metric='euclidean')
        elif self.distance_mode == 'correlation':
            geo1 = pairwise_distances(centers1, metric='correlation')
            geo2 = pairwise_distances(centers2, metric='correlation')
        elif self.distance_mode == 'RBF_kernel':
            geo1 = rbf_kernel(centers1)
            geo2 = rbf_kernel(centers2)
        else:
            raise Exception("distance_mode error! Enter a correct distance_mode.")
        
        geo1 = torch.tensor(geo1, device=device, dtype=torch.float32)
        geo2 = torch.tensor(geo2, device=device, dtype=torch.float32)
        
        return geo1, geo2

    def cal_geo_sketch(self, data1, data2, n_kmeans, device):
        res = faiss.StandardGpuResources()
        if device == 'cpu':
            index = faiss.GpuIndexFlatL2(res, data1.shape[1])
        else:
            gpu_id = int(device.split(':')[-1])
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.device = gpu_id
            index = faiss.GpuIndexFlatL2(res, data1.shape[1], flat_config)

        kmeans = faiss.Clustering(data1.shape[1], n_kmeans)
        kmeans.niter = 30
        kmeans.train(data1, index)
        centers1 = faiss.vector_to_array(kmeans.centroids).reshape(n_kmeans, data1.shape[1])
        _, assignments = index.search(data1, 1)
        ids_1 = assignments.flatten()
        
        kmeans = faiss.Clustering(data2.shape[1], n_kmeans)
        kmeans.niter = 30
        kmeans.train(data2, index)
        centers2 = faiss.vector_to_array(kmeans.centroids).reshape(n_kmeans, data2.shape[1])
        _, assignments = index.search(data2, 1)
        ids_2 = assignments.flatten()
        
        ids_1 = torch.tensor(ids_1, device=device, dtype=torch.long)
        ids_2 = torch.tensor(ids_2, device=device, dtype=torch.long)
        
        one_hot_c1 = torch.nn.functional.one_hot(ids_1, num_classes=n_kmeans).float()
        one_hot_c2 = torch.nn.functional.one_hot(ids_2, num_classes=n_kmeans).float()
        one_hot_c1_mean = one_hot_c1 / one_hot_c1.sum(dim=0, keepdim=True)
        one_hot_c2_mean = one_hot_c2 / one_hot_c2.sum(dim=0, keepdim=True)
        one_hot_c1_mean = one_hot_c1_mean.to(device)
        one_hot_c2_mean = one_hot_c2_mean.to(device)
        
        return centers1, centers2, one_hot_c1_mean, one_hot_c2_mean
        
    def split_find_correspondece(
        self,
        n_gpus,
        data1,
        data2,
        coord=None,
        epoch=2000,
        lambda_e=0,
        lambda_n=0.001,
        lr=1e-2,
        integration_mode='h',
        use_topology=True,
        geo_sketch=True,
        n_kmeans=1000,
        use_COVET=False
    ):
        
        indices = np.arange(data1.shape[0])
        np.random.shuffle(indices)
        data1_split = np.array_split(data1[indices], n_gpus, axis=0)
        if coord is not None:
            coord_split = np.array_split(coord[indices], n_gpus, axis=0)
        manager = mp.Manager()
        F_list = manager.list()
        b_list = manager.list()
        
        n_split_k = 1
        if n_gpus > torch.cuda.device_count():
            # split data1_split into n_split // torch.cuda.device_count() parts
            n_split_k = math.ceil(n_gpus/torch.cuda.device_count())
            n_gpus = torch.cuda.device_count()
        
        for k in range(n_split_k):
            processes = []
            n = n_gpus * k
            for i in range(n, n_gpus+n):
                print('Begin Split: {}'.format(i))
                data1_chunk = data1_split[i]
                if coord is not None:
                    coord_chunk = coord_split[i]
                else:
                    coord_chunk = None
                p = mp.Process(
                    target=self._process_split,
                    args=(data1_chunk, 
                          coord_chunk, 
                          data2, epoch, 
                          lambda_e, 
                          lambda_n, 
                          lr, 
                          integration_mode, 
                          use_topology, 
                          geo_sketch, 
                          n_kmeans, 
                          i, 
                          F_list, 
                          b_list,
                          use_COVET
                        )
                )
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

        F_list = sorted(F_list, key=lambda x: x[0])
        F = np.concatenate([f[1] for f in F_list], axis=0)
        b_list = sorted(b_list, key=lambda x: x[0])
        b = np.concatenate([b[1] for b in b_list], axis=0)
        return F[np.argsort(indices)], b[np.argsort(indices)]

    def _process_split(
        self,
        data1_chunk,
        coord_chunk,
        data2,
        epoch,
        lambda_e,
        lambda_n,
        lr,
        integration_mode,
        use_topology,
        geo_sketch,
        n_kmeans,
        index,
        F_list,
        b_list,
        use_COVET
    ):
        device = f'cuda:{index % torch.cuda.device_count()}'
        F_i, b_i = self.find_correspondece(
            data1=data1_chunk,
            coord=coord_chunk,
            data2=data2,
            epoch=epoch,
            lambda_e=lambda_e,
            lambda_n=lambda_n,
            lr=lr,
            integration_mode=integration_mode,
            use_topology=use_topology,
            geo_sketch=geo_sketch,
            n_kmeans=n_kmeans,
            device=device,
            use_COVET=use_COVET
        )
        F_list.append((index, F_i))
        b_list.append((index, b_i))
            
    def find_correspondece(
        self, 
        data1,  # spatial data
        data2,  # single-cell data
        coord=None,
        epoch=1000, 
        lambda_e=0, 
        lambda_n=0.001,
        lambda_geo=1,
        lr=0.01, 
        integration_mode='h', 
        use_topology=True,
        geo_sketch=True, 
        n_kmeans=1000,
        device='cpu',
        use_COVET=False,
        mode='cell'
    ):
        
        distance_modes =  ['euclidean', 'correlation', 'geodesic', 'RBF_kernel']

        if self.distance_mode != 'geodesic' and self.distance_mode not in distance_modes:
                raise Exception("distance_mode error! Enter a correct distance_mode.")  
            
        self.init_random_seed(self.manual_seed)
        
        if use_COVET:
            print('start calculating COVET')
            neighbor_indices = self.cal_data_with_neighbors(coord, k=20)
            data = torch.tensor(data1, device=device, dtype=torch.float32)
            sigma = self.cal_COVET(data, neighbor_indices)
            sigma = sigma.to(device)
        
        pca = PCA(n_components=50)
        
        if (integration_mode == 'h' and use_topology and geo_sketch) or mode=='cluster':
            if data1.shape[1] > 500:
                data1_ = pca.fit_transform(data1)
            else:
                data1_ = data1
            if data2.shape[1] > 500:
                data2_ = pca.fit_transform(data2)
            else:
                data2_ = data2
            centers1, centers2, one_hot_c1_mean, one_hot_c2_mean = self.cal_geo_sketch(data1_, data2_, n_kmeans, device)
        else:
            centers1 = data1
            centers2 = data2
        
        if use_topology:
            geo1, geo2 = self.cal_self_similarity(centers1, centers2, device)
            
        data1 = torch.tensor(data1, device=device, dtype=torch.float32)
        data2 = torch.tensor(data2, device=device, dtype=torch.float32)
        
        if mode=='cell':
            self.F = torch.randn(data1.shape[0], data2.shape[0], device=device, requires_grad=True, dtype=torch.float32)     
        else:
            self.F = torch.randn(n_kmeans, n_kmeans, device=device, requires_grad=True, dtype=torch.float32)
    
        optimizer = torch.optim.Adam([self.F], lr=lr)
        
        similarity_geo = 0
        similarity_cell = 0
    
        progress = tqdm(range(epoch), desc='Score: ', leave=True)
        for ep in progress:
            
            F_probs = softmax(self.F, dim=1)
            
            if use_topology:
                if integration_mode == 'h' and geo_sketch and mode=='cell':          
                    F_reduced = one_hot_c1_mean.T @ F_probs @ one_hot_c2_mean
                else:
                    F_reduced = F_probs
                    
                geo1_ = F_reduced @ geo2 @ F_reduced.T
                similarity_geo_0 = cosine_similarity(geo1, geo1_, dim=0).mean()
                similarity_geo_1 = cosine_similarity(geo1, geo1_, dim=1).mean()
                similarity_geo = (similarity_geo_0 + similarity_geo_1) / 2
              
            if integration_mode == 'h':
                if mode=='cluster':
                    F_probs = one_hot_c1_mean @ F_probs @ one_hot_c2_mean.T
                data1_ = F_probs @ data2
                similarity_cell_0 = cosine_similarity(data1, data1_, dim=0).mean()
                similarity_cell_1 = cosine_similarity(data1, data1_, dim=1).mean()
                similarity_cell = (similarity_cell_0 + similarity_cell_1) / 2
                total_loss = -(similarity_geo * lambda_geo + similarity_cell)
            else:
                total_loss = -(similarity_geo * lambda_geo)
            
            similarity = similarity_geo + similarity_cell
            
            if use_COVET and ep % 10 == 0:
                sigma_ = self.cal_COVET(data1_, neighbor_indices)
                niche_term = torch.norm(sigma - sigma_)
                total_loss += lambda_n * niche_term

            if ep>500 and lambda_e!=0:
                # entropy_term = lambda_e * torch.log(F_probs) * F_probs
                # calculate entropy loss for each row
                entropy_term = lambda_e * torch.sum(F_probs * torch.log(F_probs), dim=1)
                total_loss -= entropy_term.mean()
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if ep % 20 == 0:
                progress.set_description('total_loss: {:.4f}, similarity: {:.4f}'.format(total_loss.item(), similarity.item()))
                
            torch.cuda.empty_cache()
            
        return F_probs.detach().cpu().numpy()
            
    def align(self, data2, F, device='cpu'):
        data2 = torch.from_numpy(data2).float().to(device)
        F = torch.from_numpy(F).float().to(device)
        data1_aligned = F @ data2
        data2_aligned = data2
        data1_aligned = data1_aligned.cpu().numpy()
        data2_aligned = data2_aligned.cpu().numpy() 
        if data2.shape[1] > 500:
            pca = PCA(n_components=30)
            data_aligned = np.concatenate((data1_aligned, data2_aligned), axis=0)
            data_aligned = pca.fit_transform(data_aligned)
            data1_aligned = data_aligned[:data1_aligned.shape[0], :]
            data2_aligned = data_aligned[data1_aligned.shape[0]:, :]
        return data1_aligned, data2_aligned
    
        
    
       