import torch 
from torch import nn

# clase que recibe una imagen, la divide en patches,
# realiza la proyeccion, añade el class embedding y el 
# position embedding
class ImgEmbedding(nn.Module):
    def __init__(self,
                 patch_size = 16, 
                 embedding_dim = 768):
        super().__init__()
        
        # se realizan convoluciones que mapean cada patch
        # individualmente al tamaño deseado en el embedding
        self.ImgPatches = nn.Conv2d(in_channels=3,
                                    out_channels=embedding_dim,
                                    kernel_size=patch_size,
                                    stride=patch_size,
                                    padding=0)

        # permite
        self.flatten = nn.Flatten(start_dim=2)
        
        # parámetro que se añadirá al comienzo de 
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim))
        
        # position embedding que se le suma a cada patch, permitiendo que el sistema
        # aprenda sobre la posicion de cada uno
        self.position_embedding = nn.Parameter(data=torch.randn(1, 197, embedding_dim))
    
    def forward(self, x):
        # se obtienen los patches de la imagen de entrada
        x = self.ImgPatches(x)
        # se aplica el flatten y se ordenan las dimensiones para
        # obtener un embedding de tamaño (batch,196,768)
        x = self.flatten(x).permute(0, 2, 1)
        
        # se añade el class embedding al comienzo, obteniendo un embedding
        # de tamaño (batch,197,768)
        class_embedding = self.class_embedding.expand(x.shape[0], -1, -1)
        x = torch.cat((class_embedding, x), dim=1)

        # se suma el position embedding
        x = self.position_embedding + x

        # se entrega la salida de tamaño (batch,197,768) que 
        # posteriormente entrara al encoder
        return x

# clase que define bloques que componen el Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self,
                 embedding_dim = 768, 
                 num_heads = 12, 
                 mlp_size = 3072): 
        super().__init__()

        # primer bloque de Laryer Normalizarion
        self.layerNorm1 = nn.LayerNorm(normalized_shape=embedding_dim)

        # bloque de Multihead Attention
        self.msa = nn.MultiheadAttention(embed_dim=embedding_dim,
                                         num_heads=num_heads,
                                         batch_first=True) 
    
        # segunda capa de Layer Normalization
        self.layerNorm2 = nn.LayerNorm(normalized_shape=embedding_dim)

        # capa MLP final del encoder, se compone de dos bloques lineales
        # con una GELU intermedia que aplica una no linealidad
        self.mlp = nn.Sequential(nn.Linear(in_features=embedding_dim,
                                           out_features=mlp_size),
                                 nn.GELU(),
                                 nn.Linear(in_features=mlp_size,
                                           out_features=embedding_dim))

    def forward(self, x):
        
        # se aplica la normalizacion de la entrada
        x_norm = self.layerNorm1(x)

        # se pasa por el mecanismo de self attention, donde 
        # el query, key y value es la misma entrada
        x_attention = self.msa(query=x_norm,  
                               key=x_norm, 
                               value=x_norm, 
                               need_weights=False)[0]
        
        # se implementa la conexion residual del encoder
        x =  x + x_attention 
        
        # se aplica la segunda Layer Normalization
        x_norm = self.layerNorm2(x)

        # se pasa por la capa MLP
        x_mlp = self.mlp(x_norm)

        # se implementa la conexion residual
        x = x + x_mlp
        
        # se devuelve la salida del encoder
        return x

# clase que agrupa todos los bloques para construir el 
# vision transformer utilizado para clasificación
class MyViT(nn.Module):
    def __init__(self,
               patch_size = 16, 
               transformer_layers = 12, 
               embedding_dim = 768, 
               mlp_size = 3072, 
               num_heads = 12, 
               num_classes = 82): 
        super().__init__() 
        
        # bloque que obtiene los embeddings que se ingresaran a los encoders
        self.patch_embedding = ImgEmbedding(patch_size=patch_size,
                                            embedding_dim=embedding_dim)
        
        # se concatenan los 12 encoder de forma secuencial
        self.transformer_encoder = nn.Sequential(*[TransformerEncoder(embedding_dim=embedding_dim,
                                                                      num_heads=num_heads,
                                                                      mlp_size=mlp_size) for _ in range(transformer_layers)])
       
        # ultima capa de la red, contiene una capa lineal que permite mapear los
        # embeddings al tamaño deseado en la salida para realizar la clasificacion
        self.mlp_head = nn.Sequential(nn.LayerNorm(normalized_shape=embedding_dim),
                                      nn.Linear(in_features=embedding_dim,
                                                out_features=num_classes)
        )
    
    def forward(self, x):
        # se obtienen los embeddings
        x = self.patch_embedding(x)
        # se pasa por los encoders para obtener la representacion latente
        x = self.transformer_encoder(x)
        # se aplica el clasificador sobre los parametros del primer patch
        x = self.mlp_head(x[:, 0]) 

        # se devuelven los resultados obtenidos para cada clase
        return x   