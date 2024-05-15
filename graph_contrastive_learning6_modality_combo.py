
"""

graph_contrastive_learning6_modality_Combo.ipynb


"""


def read_data(content,cite):
  all_data = []
  all_edges = []

  #for root,dirs,files in os.walk('./cora'):
  #   for file in files:
  #      if '.content' in file:
  with open(content,'r') as f:
      all_data.extend(f.read().splitlines())
  #elif 'cites' in file:
  with open(cite,'r') as f:
      all_edges.extend(f.read().splitlines())

  return(all_data,all_edges)



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


# Assuming you have already obtained test_embeddings in the code provided earlier

# Load the ground truth labels for test nodes
def classifyNodes(train_idx,test_idx,train_embeddings,test_embeddings,labels):
      train_labels = labels[train_idx]
      test_labels = labels[test_idx]

      
      classifier = RandomForestClassifier(n_estimators=100, random_state=42)
      classifier.fit(train_embeddings, train_labels)

      
      # Predict labels for test nodes
      predicted_labels = classifier.predict(test_embeddings)


      # Calculate accuracy
      accuracy = accuracy_score(test_labels, predicted_labels)

      print(f"Accuracy: {accuracy:.4f}")


      # Calculate sensitivity (recall)
      sensitivity = recall_score(test_labels, predicted_labels, average='binary')
      print(f"sensitivity: {sensitivity:.4f}")


      # Calculate precision
      precision = precision_score(test_labels, predicted_labels, average='binary')
      print(f"precision: {precision:.4f}")

      # Calculate F1 score
      f1 = f1_score(test_labels, predicted_labels, average='binary')
      print(f"f1: {f1:.4f}")

      # Calculate AUC value
      auc_value = roc_auc_score(test_labels, classifier.predict_proba(test_embeddings)[:, 1])

      # Assuming y_true and y_pred are your actual and predicted labels
      cm = confusion_matrix(test_labels, predicted_labels)

      # Extracting True Negative (TN) and False Positive (FP) from the confusion matrix
      TN = cm[0, 0]
      FP = cm[0, 1]

      # Calculating specificity
      specificity = TN / (TN + FP)






      return(accuracy,sensitivity, precision,f1,specificity,auc_value)

# new attention :========

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.special import softmax
def classifyNodes1(train_idx,test_idx,train_embeddings1,train_embeddings2,train_embeddings3 ,train_embeddings4,train_embeddings5,train_embeddings6, \
                   test_embeddings1,test_embeddings2 ,test_embeddings3,test_embeddings4,test_embeddings5,test_embeddings6,labels):
      class CrossModalityAttentionModel(tf.keras.Model):
          def __init__(self, input_dim, hidden_dim, output_dim, num_heads=1):
              super(CrossModalityAttentionModel, self).__init__()
              self.embedding_layer = tf.keras.layers.Dense(hidden_dim)
              self.num_heads = num_heads
              # self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_dim)
              # self.classifier = tf.keras.layers.Dense(output_dim)
              self.classifier = tf.keras.layers.Dense(output_dim,activation='sigmoid')

          def call(self, inputs):
              input1, input2, input3,input4,input5,input6 = inputs

              # Embedding each modality
              q1, k1, v1 = self.embedding_layer(input1), self.embedding_layer(input1), self.embedding_layer(input1)
              q2, k2, v2 = self.embedding_layer(input2), self.embedding_layer(input2), self.embedding_layer(input2)
              q3, k3, v3 = self.embedding_layer(input3), self.embedding_layer(input3), self.embedding_layer(input3)
              q4, k4, v4 = self.embedding_layer(input4), self.embedding_layer(input4), self.embedding_layer(input4)
              q5, k5, v5 = self.embedding_layer(input5), self.embedding_layer(input5), self.embedding_layer(input5)
              q6, k6, v6 = self.embedding_layer(input6), self.embedding_layer(input6), self.embedding_layer(input6)



              # print("q shape",q1.shape, "k shape",k1.shape, "v shape",v1.shape)

              # Cross-Modality Attention with Q, K, V dot products
              attention_weights_12 = tf.matmul(q1, tf.transpose(k2)) / tf.math.sqrt(tf.constant(hidden_dim, dtype=tf.float32))
              attention_weights_13 = tf.matmul(q1, tf.transpose(k3)) / tf.math.sqrt(tf.constant(hidden_dim, dtype=tf.float32))
              attention_weights_14 = tf.matmul(q1, tf.transpose(k4)) / tf.math.sqrt(tf.constant(hidden_dim, dtype=tf.float32))
              attention_weights_15 = tf.matmul(q1, tf.transpose(k5)) / tf.math.sqrt(tf.constant(hidden_dim, dtype=tf.float32))
              attention_weights_16 = tf.matmul(q1, tf.transpose(k6)) / tf.math.sqrt(tf.constant(hidden_dim, dtype=tf.float32))

              attention_weights_21 = tf.matmul(q2, tf.transpose(k1)) / tf.math.sqrt(tf.constant(hidden_dim, dtype=tf.float32))
              attention_weights_23 = tf.matmul(q2, tf.transpose(k3)) / tf.math.sqrt(tf.constant(hidden_dim, dtype=tf.float32))
              attention_weights_24 = tf.matmul(q2, tf.transpose(k4)) / tf.math.sqrt(tf.constant(hidden_dim, dtype=tf.float32))
              attention_weights_25 = tf.matmul(q2, tf.transpose(k5)) / tf.math.sqrt(tf.constant(hidden_dim, dtype=tf.float32))
              attention_weights_26 = tf.matmul(q2, tf.transpose(k6)) / tf.math.sqrt(tf.constant(hidden_dim, dtype=tf.float32))

              attention_weights_31 = tf.matmul(q3, tf.transpose(k1)) / tf.math.sqrt(tf.constant(hidden_dim, dtype=tf.float32))
              attention_weights_32 = tf.matmul(q3, tf.transpose(k2)) / tf.math.sqrt(tf.constant(hidden_dim, dtype=tf.float32))
              attention_weights_34 = tf.matmul(q3, tf.transpose(k4)) / tf.math.sqrt(tf.constant(hidden_dim, dtype=tf.float32))
              attention_weights_35 = tf.matmul(q3, tf.transpose(k5)) / tf.math.sqrt(tf.constant(hidden_dim, dtype=tf.float32))
              attention_weights_36 = tf.matmul(q3, tf.transpose(k6)) / tf.math.sqrt(tf.constant(hidden_dim, dtype=tf.float32))

              attention_weights_41 = tf.matmul(q4, tf.transpose(k1)) / tf.math.sqrt(tf.constant(hidden_dim, dtype=tf.float32))
              attention_weights_42 = tf.matmul(q4, tf.transpose(k2)) / tf.math.sqrt(tf.constant(hidden_dim, dtype=tf.float32))
              attention_weights_43 = tf.matmul(q4, tf.transpose(k3)) / tf.math.sqrt(tf.constant(hidden_dim, dtype=tf.float32))
              attention_weights_45 = tf.matmul(q4, tf.transpose(k5)) / tf.math.sqrt(tf.constant(hidden_dim, dtype=tf.float32))
              attention_weights_46 = tf.matmul(q4, tf.transpose(k6)) / tf.math.sqrt(tf.constant(hidden_dim, dtype=tf.float32))

              attention_weights_51 = tf.matmul(q5, tf.transpose(k1)) / tf.math.sqrt(tf.constant(hidden_dim, dtype=tf.float32))
              attention_weights_52 = tf.matmul(q5, tf.transpose(k2)) / tf.math.sqrt(tf.constant(hidden_dim, dtype=tf.float32))
              attention_weights_53 = tf.matmul(q5, tf.transpose(k3)) / tf.math.sqrt(tf.constant(hidden_dim, dtype=tf.float32))
              attention_weights_54 = tf.matmul(q5, tf.transpose(k4)) / tf.math.sqrt(tf.constant(hidden_dim, dtype=tf.float32))
              attention_weights_56 = tf.matmul(q5, tf.transpose(k6)) / tf.math.sqrt(tf.constant(hidden_dim, dtype=tf.float32))

              attention_weights_61 = tf.matmul(q6, tf.transpose(k1)) / tf.math.sqrt(tf.constant(hidden_dim, dtype=tf.float32))
              attention_weights_62 = tf.matmul(q6, tf.transpose(k2)) / tf.math.sqrt(tf.constant(hidden_dim, dtype=tf.float32))
              attention_weights_63 = tf.matmul(q6, tf.transpose(k3)) / tf.math.sqrt(tf.constant(hidden_dim, dtype=tf.float32))
              attention_weights_64 = tf.matmul(q6, tf.transpose(k4)) / tf.math.sqrt(tf.constant(hidden_dim, dtype=tf.float32))
              attention_weights_65 = tf.matmul(q6, tf.transpose(k5)) / tf.math.sqrt(tf.constant(hidden_dim, dtype=tf.float32))

              # Calculate attention outputs
              output1 = tf.matmul(attention_weights_12, v2) + tf.matmul(attention_weights_13, v3)+ tf.matmul(attention_weights_14, v4)+tf.matmul(attention_weights_15, v5)+tf.matmul(attention_weights_16, v6)
              output2 = tf.matmul(attention_weights_21, v1) + tf.matmul(attention_weights_23, v3) + tf.matmul(attention_weights_24, v4)+tf.matmul(attention_weights_25, v5)+tf.matmul(attention_weights_26, v6)
              output3 = tf.matmul(attention_weights_31, v1) + tf.matmul(attention_weights_32, v2)+ tf.matmul(attention_weights_34, v4)+tf.matmul(attention_weights_35, v5)+tf.matmul(attention_weights_36, v6)
              output4 = tf.matmul(attention_weights_41, v1) + tf.matmul(attention_weights_42, v2)+ tf.matmul(attention_weights_43, v3)+tf.matmul(attention_weights_45, v5)+tf.matmul(attention_weights_46, v6)
              output5 = tf.matmul(attention_weights_51, v1) + tf.matmul(attention_weights_52, v2)+ tf.matmul(attention_weights_53, v3)+tf.matmul(attention_weights_54, v4)+tf.matmul(attention_weights_56, v6)
              output6 = tf.matmul(attention_weights_61, v1) + tf.matmul(attention_weights_62, v2)+ tf.matmul(attention_weights_63, v3)+tf.matmul(attention_weights_64, v4)+tf.matmul(attention_weights_65, v5)


              # Combine attention outputs across modalities
              combined_output = output1 + output2 + output3 + output4+ output5 +output6

              # Classification
              output = self.classifier(combined_output)
              # print(combined_output )
              return(output,combined_output)

      # Example data (random tensors)
      input_dim = 32 #128
      # num_samples = 1000

      # Create an instance of the model
      hidden_dim =16 #64
      output_dim = 1  # Binary classification
      num_heads = 4 #8 16 # Number of attention heads
      model = CrossModalityAttentionModel(input_dim, hidden_dim, output_dim, num_heads)



      # Define loss function and optimizer
      criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)  # Binary cross-entropy loss
      optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


      input1_train= train_embeddings1
      input1_test=test_embeddings1
      input2_train= train_embeddings2
      input2_test=test_embeddings2
      input3_train=train_embeddings3
      input3_test=test_embeddings3
      input4_train=train_embeddings4
      input4_test=test_embeddings4
      input5_train=train_embeddings5
      input5_test=test_embeddings5
      input6_train=train_embeddings6
      input6_test=test_embeddings6
      targets_train=labels[train_idx]
      targets_test=labels[test_idx]

      samples,feaatures=train_embeddings1.shape
      num_samples=samples

      # Training loop
      num_epochs = 50 #50#10
      batch_size = 32 #32

      for epoch in range(num_epochs):
          for i in range(0, num_samples, batch_size):
              input1_batch, input2_batch, input3_batch,input4_batch,input5_batch,input6_batch= input1_train[i:i+batch_size],\
                             input2_train[i:i+batch_size], input3_train[i:i+batch_size],input4_train[i:i+batch_size],input5_train[i:i+batch_size],input6_train[i:i+batch_size]
              targets_batch = targets_train[i:i+batch_size]

              # Forward pass
              with tf.GradientTape() as tape:
                  outputs,combined_output = model([input1_batch, input2_batch, input3_batch, input4_batch, input5_batch, input6_batch], training=True)
                  loss = criterion(targets_batch, outputs)

              # Backward pass and optimization
              gradients = tape.gradient(loss, model.trainable_variables)
              optimizer.apply_gradients(zip(gradients, model.trainable_variables))

          # Print loss
          print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')

      # Testing the model


      # Get predictions using model.predict
      predictions_test, intermediate_representation_test = model.predict([input1_test, input2_test, input3_test,input4_test,input5_test,input6_test], batch_size=batch_size)

      # intermediate_representation = model([input1_test, input2_test, input3_test])
      print("intermediate_representation test shape=====",intermediate_representation_test.shape)

      predictions_train, intermediate_representation_train = model.predict([input1_train, input2_train, input3_train,input4_train,input5_train,input6_train], batch_size=batch_size)
      print("intermediate_representation train shape=====",intermediate_representation_train.shape)

      accuracy,sensitivity, precision,f1,specificity,auc_value=classifyNodes(train_idx,test_idx,intermediate_representation_train,intermediate_representation_test,labels)

      return(accuracy,sensitivity, precision,f1,specificity,auc_value)

import dgl
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from dgl.data import CoraGraphDataset
from dgl.nn import GraphConv
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dgl.data import DGLDataset
import sys
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from itertools import combinations


def CustomGraphDataset(content1,content2,content3,content4,content5,content6,cite1,cite2,cite3,cite4,cite5,cite6):

        df=pd.read_csv(content1,header=None)
        labels = df.iloc[:, -1].tolist()
        df = df.iloc[:, 1:]
        df = df.drop(df.columns[-1], axis=1)

        df1 = pd.read_csv(content2,header=None)
        df1 = df1.iloc[:, 1:]
        df1 = df1.drop(df1.columns[-1], axis=1)

        df2 = pd.read_csv(content3,header=None)
        df2 = df2.iloc[:, 1:]
        df2 = df2.drop(df2.columns[-1], axis=1)

        df3 = pd.read_csv(content4,header=None)
        df3 = df3.iloc[:, 1:]
        df3 = df3.drop(df3.columns[-1], axis=1)

        df4 = pd.read_csv(content5,header=None)
        df4 = df4.iloc[:, 1:]
        df4 = df4.drop(df4.columns[-1], axis=1)

        df5 = pd.read_csv(content6,header=None)
        df5 = df5.iloc[:, 1:]
        df5 = df5.drop(df5.columns[-1], axis=1)

        cite = pd.read_csv(cite1,delimiter='\t',header=None)
        cite1 = pd.read_csv(cite2,delimiter='\t',header=None)
        cite2 = pd.read_csv(cite3,delimiter='\t',header=None)
        cite3 = pd.read_csv(cite4,delimiter='\t',header=None)
        cite4 = pd.read_csv(cite5,delimiter='\t',header=None)
        cite5 = pd.read_csv(cite6,delimiter='\t',header=None)


        last_column_name = df.columns[-1]

        print ( df[last_column_name].values)
        last_column_name = df1.columns[-1]
        print ( df1[last_column_name].values)
        print ( df2[last_column_name].values)
        print ( df3[last_column_name].values)
        print ( df4[last_column_name].values)
        last_column_name = df5.columns[-1]
        print ( df5[last_column_name].values)


        #  Read two columns separately by index
        column1_index = 0  # Index of 'Column1'
        column2_index = 1  # Index of 'Column2'

        column1_data = cite.iloc[:, column1_index].tolist()
        column2_data = cite.iloc[:, column2_index].tolist()


        src = list(map(lambda x: x - 1, column1_data))
        dst=list(map(lambda x: x - 1, column2_data))


        # Create a DGL graph
        nodes,features= df.shape

        num_nodes=nodes
        g1 = dgl.graph((src, dst), num_nodes=num_nodes)
        # Add self-loops to the graph
        g1 = dgl.add_self_loop(g1)
        tensor_data = torch.tensor(df.values, dtype=torch.float32)


        labels = torch.tensor(labels)
        g1.ndata['feat'] =tensor_data
        # return([g1],labels)

        #=============================================================================


        column1_index1 = 0  # Index of 'Column1'
        column2_index1 = 1  # Index of 'Column2'

        column1_data1 = cite1.iloc[:, column1_index1].tolist()
        column2_data1 = cite1.iloc[:, column2_index1].tolist()


        src1 = list(map(lambda x: x - 1, column1_data1))
        dst1 = list(map(lambda x: x - 1, column2_data1))

        # Create a DGL graph
        nodes1,features1= df1.shape

        num_nodes1=nodes1
        g2 = dgl.graph((src1, dst1), num_nodes=num_nodes1)
        # Add self-loops to the graph
        g2 = dgl.add_self_loop(g2)
        tensor_data1 = torch.tensor(df1.values, dtype=torch.float32)
        g2.ndata['feat'] =tensor_data1
        #=============================================================================
        column1_index1 = 0  # Index of 'Column1'
        column2_index1 = 1  # Index of 'Column2'

        column1_data1 = cite2.iloc[:, column1_index1].tolist()
        column2_data1 = cite2.iloc[:, column2_index1].tolist()


        src2 = list(map(lambda x: x - 1, column1_data1))
        dst2 = list(map(lambda x: x - 1, column2_data1))

        # Create a DGL graph
        nodes2,features2= df2.shape

        num_nodes1=nodes2
        g3 = dgl.graph((src2, dst2), num_nodes=num_nodes1)
        # Add self-loops to the graph
        g3 = dgl.add_self_loop(g3)
        tensor_data2 = torch.tensor(df2.values, dtype=torch.float32)
        g3.ndata['feat'] =tensor_data2

        #=============================================================================
        column1_index1 = 0  # Index of 'Column1'
        column2_index1 = 1  # Index of 'Column2'

        column1_data1 = cite3.iloc[:, column1_index1].tolist()
        column2_data1 = cite3.iloc[:, column2_index1].tolist()


        src3 = list(map(lambda x: x - 1, column1_data1))
        dst3 = list(map(lambda x: x - 1, column2_data1))

        # Create a DGL graph
        nodes3,features3= df3.shape

        num_nodes1=nodes3
        g4 = dgl.graph((src3, dst3), num_nodes=num_nodes1)
        # Add self-loops to the graph
        g4 = dgl.add_self_loop(g4)
        tensor_data3 = torch.tensor(df3.values, dtype=torch.float32)
        g4.ndata['feat'] =tensor_data3
#=============================================================================
        column1_index1 = 0  # Index of 'Column1'
        column2_index1 = 1  # Index of 'Column2'

        column1_data1 = cite4.iloc[:, column1_index1].tolist()
        column2_data1 = cite4.iloc[:, column2_index1].tolist()


        src4 = list(map(lambda x: x - 1, column1_data1))
        dst4 = list(map(lambda x: x - 1, column2_data1))

        # Create a DGL graph
        nodes4,features4= df4.shape

        num_nodes1=nodes4
        g5 = dgl.graph((src4, dst4), num_nodes=num_nodes1)
        # Add self-loops to the graph
        g5 = dgl.add_self_loop(g5)
        tensor_data4 = torch.tensor(df4.values, dtype=torch.float32)
        g5.ndata['feat'] =tensor_data4

#=============================================================================
        column1_index1 = 0  # Index of 'Column1'
        column2_index1 = 1  # Index of 'Column2'

        column1_data1 = cite5.iloc[:, column1_index1].tolist()
        column2_data1 = cite5.iloc[:, column2_index1].tolist()


        src5 = list(map(lambda x: x - 1, column1_data1))
        dst5 = list(map(lambda x: x - 1, column2_data1))

        # Create a DGL graph
        nodes5,features5= df5.shape

        num_nodes1=nodes5
        g6 = dgl.graph((src5, dst5), num_nodes=num_nodes1)
        # Add self-loops to the graph
        g6 = dgl.add_self_loop(g6)
        tensor_data5 = torch.tensor(df5.values, dtype=torch.float32)
        g6.ndata['feat'] =tensor_data5

        # list_sum = sum(labels)
        # print("labels=========",list_sum )

        return([g1],[g2],[g3],[g4],[g5],[g6],labels)






class GCNContrastive(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GCNContrastive, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats)
        self.conv2 = GraphConv(hidden_feats, out_feats)
        self.final_linear = nn.Linear(out_feats, 2)

    def forward(self, g1, x):
        x = x.float()
        first_Layer = torch.relu(self.conv1(g1, x))
        hidden_layer = self.conv2(g1, first_Layer)
        x = self.final_linear(hidden_layer)  # Additional linear layer
        return x,hidden_layer





# Create a DataLoader for positive pairs (similar nodes)
class PositivePairDataset(torch.utils.data.Dataset):
    def __init__(self, graph, nodes, transform=None):
        self.graph = graph
        self.nodes = nodes
        self.transform = transform
        self.positive_pairs = self.generate_positive_pairs()

    def generate_positive_pairs(self):
        positive_pairs = []
        for node1 in self.nodes:
            positive_candidates = np.where(self.graph.has_edges_between(node1, self.nodes))[0]

            if len(positive_candidates) > 0:
                node2 = np.random.choice(positive_candidates)
                positive_pairs.append((node1, node2))
        return positive_pairs

    def __len__(self):
        return len(self.positive_pairs)

    def __getitem__(self, idx):
        node1, node2 = self.positive_pairs[idx]
        return node1, node2


# Define a DataLoader for negative pairs (dissimilar nodes)
class NegativePairDataset(torch.utils.data.Dataset):
    def __init__(self, graph, nodes, transform=None):
        self.graph = graph
        self.nodes = nodes
        self.transform = transform
        self.negative_pairs = self.generate_negative_pairs()

    def generate_negative_pairs(self):
        negative_pairs = []
        for node1 in self.nodes:
            negative_candidates = np.where(~self.graph.has_edges_between(node1, self.nodes))[0]
            if len(negative_candidates) > 0:
                node2 = np.random.choice(negative_candidates)
                negative_pairs.append((node1, node2))
        return negative_pairs

    def __len__(self):
        return len(self.negative_pairs)

    def __getitem__(self, idx):
        node1, node2 = self.negative_pairs[idx]
        return node1, node2

#========================start main code for data reading and funtion calling================
#============================================================================================

files_content = ["/content/drive/MyDrive/colab data/6mod breast cancer/withIndexNew/file_cln.csv",\
         "/content/drive/MyDrive/colab data/6mod breast cancer/withIndexNew/file_cnv.csv",\
         "/content/drive/MyDrive/colab data/6mod breast cancer/withIndexNew/file_dna.csv",\
         "/content/drive/MyDrive/colab data/6mod breast cancer/withIndexNew/file_mir.csv",\
         "/content/drive/MyDrive/colab data/6mod breast cancer/withIndexNew/file_mrna.csv",\
                 "/content/drive/MyDrive/colab data/6mod breast cancer/withIndexNew/file_wsi.csv"
         ]

files_cite= ["/content/drive/MyDrive/colab data/6mod breast cancer/withIndexNew/edges/cln0.9_edges.cites",\
              "/content/drive/MyDrive/colab data/6mod breast cancer/withIndexNew/edges/cnv_edges.cites",\
              "/content/drive/MyDrive/colab data/6mod breast cancer/withIndexNew/edges/DNA_edges.cites",\
              "/content/drive/MyDrive/colab data/6mod breast cancer/withIndexNew/edges/mir_edges.cites",\
              "/content/drive/MyDrive/colab data/6mod breast cancer/withIndexNew/edges/mrna_edges.cites",\
             "/content/drive/MyDrive/colab data/6mod breast cancer/withIndexNew/edges/wsi_edges.cites"

              ]

# Get all permutations of [1, 2, 3]
list_content = combinations(files_content, 6)
list_cite=combinations(files_cite, 6)

file1 = open("/content/drive/MyDrive/colab data/6mod breast cancer/withIndexNew/result/output_6mod_contrast_GCN.csv","w")
# file1 = open("/content/drive/MyDrive/colab data/6mod breast cancer/withIndexNew/output_6mod_non_contrast_GCN.csv","w")


all_avg_acc=0
all_avg_sen=0
all_avg_pre=0
all_avg_f1_val=0
all_avg_spe=0
all_avg_auc=0

count=0

for combo_content,combo__cite in zip(list(list_content), list(list_cite)):
    count =count+1
    print("combination count===================================",count)

    #auc_value=0
    acc=0
    sen=0
    pre=0
    f1_val=0
    spe=0
    auc=0
    acc_l=[]
    sen_l=[]
    pre_l=[]
    f1_l=[]
    spe_l=[]
    auc_l=[]


    content1=combo_content[0]
    content2=combo_content[1]
    content3=combo_content[2]
    content4=combo_content[3]
    content5=combo_content[4]
    content6=combo_content[5]
    cite1=combo__cite[0]
    cite2=combo__cite[1]
    cite3=combo__cite[2]
    cite4=combo__cite[3]
    cite5=combo__cite[4]
    cite6=combo__cite[5]
    print("content1===",content1)
    print("content2===",content2)
    print("content3===",content3)
    print("content4===",content4)
    print("content5===",content5)
    print("content6===",content6)




    custom_dataset1,custom_dataset2,custom_dataset3,custom_dataset4,custom_dataset5,custom_dataset6,labels = CustomGraphDataset(content1=content1, content2=content2, content3=content3,content4=content4,\
                                                                                                                                content5=content5,content6=content6,cite1=cite1, cite2=cite2,cite3=cite3,cite4=cite4,cite5=cite5,cite6=cite6)
    g1 = custom_dataset1[0]
    g2 = custom_dataset2[0]
    g3 = custom_dataset3[0]
    g4 = custom_dataset4[0]
    g5 = custom_dataset5[0]
    g6 = custom_dataset6[0]
    src, dst= g1.edges()
    src2, dst2 = g2.edges()
    src3, dst3 = g3.edges()
    src4, dst4 = g4.edges()
    src5, dst5 = g5.edges()
    src6, dst6 = g6.edges()


    print(g1,g2,g3,g4,g5,g6)

    model = GCNContrastive(g1.ndata['feat'].shape[1], 128, 32)
    model1 = GCNContrastive(g2.ndata['feat'].shape[1], 128, 32)
    model2 = GCNContrastive(g3.ndata['feat'].shape[1], 128, 32)
    model3 = GCNContrastive(g4.ndata['feat'].shape[1], 128, 32)
    model4 = GCNContrastive(g5.ndata['feat'].shape[1], 128, 32)
    model5 = GCNContrastive(g6.ndata['feat'].shape[1], 128, 32)
    contrastive_loss = nn.CosineEmbeddingLoss()

   

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
    optimizer3 = torch.optim.Adam(model3.parameters(), lr=0.001)
    optimizer4 = torch.optim.Adam(model4.parameters(), lr=0.001)
    optimizer5 = torch.optim.Adam(model5.parameters(), lr=0.001)

    file1.write(str(content1))
    file1.write("\n")
    file1.write(str(content2))
    file1.write("\n")
    file1.write(str(content3))
    file1.write("\n")
    file1.write(str(content4))
    file1.write("\n")
    file1.write(str(content5))
    file1.write("\n")
    file1.write(str(content6))
    file1.write("\n")


    temp=pd.read_csv('/content/drive/MyDrive/colab data/6mod breast cancer/withIndexNew/file_cln.csv')
    array3 = temp.values
    X_clinical= array3[:,0:-1]
    y_clinical = array3[:,-1]
    num_folds = 10
    # kf=StratifiedKFold(n_splits=no_of_fold, random_state=None, shuffle=True)
    # kf = KFold(n_splits=num_folds, random_state=42, shuffle=True)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    # ================================================================================

    # Split data into 10 folds and store the indices
    Fold=0
    # for train_idx, test_idx in kf.split(range(len(g1.nodes()))):
    for train_idx, test_idx in kf.split(X_clinical, y_clinical):

          Fold=Fold+1
          print("fold==== ",Fold)

          #========================= train 1st modality========================================================
          positive_pairs_dataset = PositivePairDataset(g1, train_idx)
          positive_pairs_loader = DataLoader(positive_pairs_dataset, batch_size=64, shuffle=True)

          # Modify the training loop to include negative pairs
          negative_pairs_dataset = NegativePairDataset(g1, train_idx)
          negative_pairs_loader = DataLoader(negative_pairs_dataset, batch_size=64, shuffle=True)

          # Training loop
          num_epochs = 3 # 10
          for epoch in range(num_epochs):
              model.train()
              # for node1, node2 in positive_pairs_loader:
              for (node1_pos, node2_pos), (node1_neg, node2_neg) in zip(positive_pairs_loader, negative_pairs_loader):
                  # for itr in range(0,10):
                  optimizer.zero_grad()

            


                  # Obtain embeddings for positive pairs
                  # emb1_pos1,hidden_layer=model(g1, g1.ndata['feat'].double())[node1_pos]
                  output1,hidden_layer1=model(g1, g1.ndata['feat'].double())
                  emb1_pos1=hidden_layer1[node1_pos]
                  emb2_pos1=hidden_layer1[node2_pos]
                  emb1_neg1=hidden_layer1[node1_neg]
                  emb2_neg1=hidden_layer1[node2_neg]
               

                  # Compute contrastive loss for positive and negative pairs
                  target_pos = torch.ones(node1_pos.shape[0])
                  target_neg = torch.zeros(node1_neg.shape[0])
                  loss_pos = contrastive_loss(emb1_pos1, emb2_pos1, target_pos)
                  loss_neg = contrastive_loss(emb1_neg1, emb2_neg1, target_neg)

                  # Total loss combines losses from positive and negative pairs
                  loss = loss_pos + loss_neg

        



                  if(Fold==1):
                    old_labels=labels
                  #loss1 = nn.CrossEntropyLoss()(outputs[train_idx], old_labels[train_idx])
                  loss1 = nn.CrossEntropyLoss()(output1[train_idx], old_labels[train_idx])
           

                  loss =loss +loss1       
                  loss.backward()
                  optimizer.step()
                  print("1st mod loss", loss)

             
          model.eval()



          #========================= train 2nd modality========================================================
          positive_pairs_dataset1 = PositivePairDataset(g2, train_idx)
          positive_pairs_loader1 = DataLoader(positive_pairs_dataset1, batch_size=64, shuffle=True)

          # Modify the training loop to include negative pairs
          negative_pairs_dataset1 = NegativePairDataset(g2, train_idx)
          negative_pairs_loader1 = DataLoader(negative_pairs_dataset1, batch_size=64, shuffle=True)


          # Training loop
          # num_epochs =  5 #10 #10
          for epoch in range(num_epochs):
              model1.train()
              # for node1, node2 in positive_pairs_loader1:
              for (node1_pos2, node2_pos2), (node1_neg2, node2_neg2) in zip(positive_pairs_loader1, negative_pairs_loader1):
                  # for itr in range(0,10):
                  optimizer1.zero_grad()
                  outputs2,hidden_layer2=model1(g2, g2.ndata['feat'].double())
                  emb1_pos2=hidden_layer2[node1_pos2]
                  emb2_pos2=hidden_layer2[node2_pos2]
                  emb1_neg2=hidden_layer2[node1_neg2]
                  emb2_neg2=hidden_layer2[node2_neg2]


                  # Compute contrastive loss for positive and negative pairs
                  target_pos2 = torch.ones(node1_pos2.shape[0])
                  target_neg2 = torch.zeros(node1_neg2.shape[0])
                  loss_pos2 = contrastive_loss(emb1_pos2, emb2_pos2, target_pos2)
                  loss_neg2 = contrastive_loss(emb1_neg2, emb2_neg2, target_neg2)
                  loss = loss_pos2 + loss_neg2

                  # cross entropy loss for last layer node embedding for GCN                 
                  loss2 = nn.CrossEntropyLoss()(outputs2[train_idx], old_labels[train_idx])
                  loss =loss +loss2
                  #new loss to remove contrastive loss
                  # loss=loss2


                  # Backpropagation and optimization step
                  loss.backward()
                  optimizer1.step()
                  print("2nd mod loss", loss)

              # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
              print("loss",loss)

          # Evaluate the learned embeddings on a downstream task (e.g., node classification)

          model1.eval()
          #========================= train 3rd modality========================================================
          positive_pairs_dataset2 = PositivePairDataset(g3, train_idx)
          positive_pairs_loader2 = DataLoader(positive_pairs_dataset2, batch_size=64, shuffle=True)

          # Modify the training loop to include negative pairs
          negative_pairs_dataset2 = NegativePairDataset(g3, train_idx)
          negative_pairs_loader2 = DataLoader(negative_pairs_dataset2, batch_size=64, shuffle=True)


          # Training loop
          # num_epochs =  5 #10 #10
          for epoch in range(num_epochs):
              model2.train()
              # for node1, node2 in positive_pairs_loader1:
              for (node1_pos3, node2_pos3), (node1_neg3, node2_neg3) in zip(positive_pairs_loader2, negative_pairs_loader2):
                  # for itr in range(0,10):
                  optimizer2.zero_grad()
                  outputs3,hidden_layer3=model2(g3, g3.ndata['feat'].double())
                  emb1_pos3=hidden_layer3[node1_pos3]
                  emb2_pos3 =hidden_layer3[node2_pos3]
                  emb1_neg3=hidden_layer3[node1_neg3]
                  emb2_neg3=hidden_layer3[node2_neg3]

                
                  # Compute contrastive loss for positive and negative pairs
                  target_pos3 = torch.ones(node1_pos3.shape[0])
                  target_neg3 = torch.zeros(node1_neg3.shape[0])
                  loss_pos3 = contrastive_loss(emb1_pos3, emb2_pos3, target_pos3)
                  loss_neg3 = contrastive_loss(emb1_neg3, emb2_neg3, target_neg3)
                  loss = loss_pos3 + loss_neg3

                  # cross entropy loss for last layer node embedding for GCN                 
                  loss3 = nn.CrossEntropyLoss()(outputs3[train_idx], old_labels[train_idx])
                  loss =loss +loss3
                  #new loss to remove contrastive loss
                 
                  # Backpropagation and optimization step
                  loss.backward()
                  optimizer2.step()
                  print("3rd mod loss", loss)

             
              print("loss",loss)

         

          model2.eval()

          #========================= train 4th modality========================================================
          positive_pairs_dataset3 = PositivePairDataset(g4, train_idx)
          positive_pairs_loader3 = DataLoader(positive_pairs_dataset3, batch_size=64, shuffle=True)

          # Modify the training loop to include negative pairs
          negative_pairs_dataset3 = NegativePairDataset(g4, train_idx)
          negative_pairs_loader3 = DataLoader(negative_pairs_dataset3, batch_size=64, shuffle=True)


          # Training loop
          # num_epochs =  5 #10 #10
          for epoch in range(num_epochs):
              model3.train()
              # for node1, node2 in positive_pairs_loader1:
              for (node1_pos4, node2_pos4), (node1_neg4, node2_neg4) in zip(positive_pairs_loader3, negative_pairs_loader3):
                  # for itr in range (0,10):
                  optimizer3.zero_grad()
                  outputs4,hidden_layer4=model3(g4, g4.ndata['feat'].double())
                  emb1_pos4=hidden_layer4[node1_pos4]
                  emb2_pos4=hidden_layer4[node2_pos4]
                  emb1_neg4=hidden_layer4[node1_neg4]
                  emb2_neg4=hidden_layer4[node2_neg4]

              

                  # Compute contrastive loss for positive and negative pairs
                  target_pos4 = torch.ones(node1_pos4.shape[0])
                  target_neg4 = torch.zeros(node1_neg4.shape[0])
                  loss_pos4 = contrastive_loss(emb1_pos4, emb2_pos4, target_pos4)
                  loss_neg4 = contrastive_loss(emb1_neg4, emb2_neg4, target_neg4)
                  loss = loss_pos4 + loss_neg4

                  # cross entropy loss for last layer node embedding for GCN               
                  loss4 = nn.CrossEntropyLoss()(outputs4[train_idx], old_labels[train_idx])
                  loss =loss +loss4
                  #new loss to remove contrastive loss
                


                  # Backpropagation and optimization step
                  loss.backward()
                  optimizer3.step()
                  print("4th mod loss", loss)

              # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
              print("loss",loss)

          # Evaluate the learned embeddings on a downstream task (e.g., node classification)

          model3.eval()
          #========================= train 5th modality========================================================
          positive_pairs_dataset4 = PositivePairDataset(g5, train_idx)
          positive_pairs_loader4 = DataLoader(positive_pairs_dataset4, batch_size=64, shuffle=True)

          # Modify the training loop to include negative pairs
          negative_pairs_dataset4 = NegativePairDataset(g5, train_idx)
          negative_pairs_loader4 = DataLoader(negative_pairs_dataset4, batch_size=64, shuffle=True)


          # Training loop
          # num_epochs =  5 #10 #10
          for epoch in range(num_epochs):
              model4.train()
              # for node1, node2 in positive_pairs_loader1:
              for (node1_pos5, node2_pos5), (node1_neg5, node2_neg5) in zip(positive_pairs_loader4, negative_pairs_loader4):
                  # for itr in range (0,10):
                  optimizer4.zero_grad()
                  outputs5,hidden_layer5=model4(g5, g5.ndata['feat'].double())
                  emb1_pos5 =hidden_layer5[node1_pos5]
                  emb2_pos5 =hidden_layer5[node2_pos5]
                  emb1_neg5=hidden_layer5[node1_neg5]
                  emb2_neg5 =hidden_layer5[node2_neg5]

                  # Compute contrastive loss for positive and negative pairs
                  target_pos5 = torch.ones(node1_pos5.shape[0])
                  target_neg5 = torch.zeros(node1_neg5.shape[0])
                  loss_pos5 = contrastive_loss(emb1_pos5, emb2_pos5, target_pos5)
                  loss_neg5 = contrastive_loss(emb1_neg5, emb2_neg5, target_neg5)
                  loss = loss_pos5 + loss_neg5

                  # cross entropy loss for last layer node embedding for GCN                 
                  loss5 = nn.CrossEntropyLoss()(outputs5[train_idx], old_labels[train_idx])
                  loss =loss +loss5
                  #new loss to remove contrastive loss
                 
                  # Backpropagation and optimization step
                  loss.backward()
                  optimizer4.step()
                  print("5th mod loss", loss)
             
              print("loss",loss)

          # Evaluate the learned embeddings on a downstream task (e.g., node classification)

          model4.eval()
          #========================= train 6th modality========================================================
          positive_pairs_dataset5 = PositivePairDataset(g6, train_idx)
          positive_pairs_loader5 = DataLoader(positive_pairs_dataset5, batch_size=64, shuffle=True)

          # Modify the training loop to include negative pairs
          negative_pairs_dataset5 = NegativePairDataset(g6, train_idx)
          negative_pairs_loader5 = DataLoader(negative_pairs_dataset5, batch_size=64, shuffle=True)


          # Training loop
          # num_epochs =  5 #10 #10
          for epoch in range(num_epochs):
              model5.train()
              # for node1, node2 in positive_pairs_loader1:
              for (node1_pos6, node2_pos6), (node1_neg6, node2_neg6) in zip(positive_pairs_loader5, negative_pairs_loader5):
                  # for itr in range (0,10):
                  optimizer5.zero_grad()
                  outputs6,hidden_layer6=model5(g6, g6.ndata['feat'].double())
                  emb1_pos6=hidden_layer6[node1_pos6]
                  emb2_pos6=hidden_layer6[node2_pos6]
                  emb1_neg6=hidden_layer6[node1_neg6]
                  emb2_neg6=hidden_layer6[node2_neg6]             


                  # Compute contrastive loss for positive and negative pairs
                  target_pos6 = torch.ones(node1_pos6.shape[0])
                  target_neg6 = torch.zeros(node1_neg6.shape[0])
                  loss_pos6 = contrastive_loss(emb1_pos6, emb2_pos6, target_pos6)
                  loss_neg6 = contrastive_loss(emb1_neg6, emb2_neg6, target_neg6)
                  loss = loss_pos6 + loss_neg6

                  # cross entropy loss for last layer node embedding for GCN
                
                  loss6 = nn.CrossEntropyLoss()(outputs6[train_idx], old_labels[train_idx])
                  loss =loss +loss6
                  
                 
                  # Backpropagation and optimization step
                  loss.backward()
                  optimizer5.step()
                  print("6th mod loss", loss)

             
              print("loss",loss)

          # Evaluate the learned embeddings on a downstream task (e.g., node classification)

          model5.eval()

          #===for clinical modality====



          def clinical_Modality(df_cln):

              # Number of columns you want after appending zeros
              desired_num_columns = 32

              # Append zeros to the DataFrame to make it 1035x128
              num_zeros_to_append = desired_num_columns - df_cln.shape[1]
              zeros_to_append = pd.DataFrame(np.zeros((df_cln.shape[0], num_zeros_to_append)), columns=range(20, desired_num_columns + 1))

              # Concatenate the original DataFrame and zeros_to_append
              expanded_dataframe = pd.concat([df_cln, zeros_to_append], axis=1)

              # Creating train set
              train_set = expanded_dataframe.iloc[train_idx]

              # Creating test set
              test_set = expanded_dataframe.iloc[test_idx]
              train_embeddings=train_set
              test_embeddings=test_set
              train_embeddings= tf.convert_to_tensor(train_embeddings)
              test_embeddings= tf.convert_to_tensor(test_embeddings)
              return(train_embeddings,test_embeddings)


          df_cln = pd.read_csv('/content/drive/MyDrive/colab data/6mod breast cancer/withIndexNew/file_cln.csv',header=None)
          df_cln = df_cln.iloc[:, 1:]
          df_cln = df_cln.drop(df_cln.columns[-1], axis=1)
          #=================================
         

          with torch.no_grad():
           

              test_embeddings1 = hidden_layer1[test_idx]
              train_embeddings1 = hidden_layer1[train_idx]

              test_embeddings2 = hidden_layer2[test_idx]
              train_embeddings2 = hidden_layer2[train_idx]
              test_embeddings3= hidden_layer3[test_idx]
              train_embeddings3 = hidden_layer3[train_idx]
              test_embeddings4= hidden_layer4[test_idx]
              train_embeddings4 = hidden_layer4[train_idx]
              test_embeddings5 = hidden_layer5[test_idx]
              train_embeddings5 = hidden_layer5[train_idx]
              test_embeddings6 = hidden_layer6[test_idx]
              train_embeddings6 = hidden_layer6[train_idx]




              flag1=0
              if(content1 =="/content/drive/MyDrive/colab data/6mod breast cancer/withIndexNew/file_cln.csv"):
                  train_embeddings1,test_embeddings1=clinical_Modality(df_cln)
                  flag1=1
              flag2=0
              if(content2 =="/content/drive/MyDrive/colab data/6mod breast cancer/withIndexNew/file_cln.csv"):
                  train_embeddings2,test_embeddings2=clinical_Modality(df_cln)
                  flag2=1
              flag3=0
              if(content3 =="/content/drive/MyDrive/colab data/6mod breast cancer/withIndexNew/file_cln.csv"):
                  train_embeddings3,test_embeddings3=clinical_Modality(df_cln)
                  flag3=1
              flag4=0
              if(content4 =="/content/drive/MyDrive/colab data/6mod breast cancer/withIndexNew/file_cln.csv"):
                  train_embeddings4,test_embeddings4=clinical_Modality(df_cln)
                  flag4=1
              flag5=0
              if(content5 =="/content/drive/MyDrive/colab data/6mod breast cancer/withIndexNew/file_cln.csv"):
                  train_embeddings5,test_embeddings5=clinical_Modality(df_cln)
                  flag5=1
              flag6=0
              if(content6 =="/content/drive/MyDrive/colab data/6mod breast cancer/withIndexNew/file_cln.csv"):
                  train_embeddings6,test_embeddings6=clinical_Modality(df_cln)
                  flag6=1



              print("flag1",flag1," flag2",flag2,"flag3",flag3,"flag4",flag4,"flag5",flag5,"flag6",flag5)

              # print(train_embeddings1.shape,train_embeddings2.shape)

              if(flag1==0):
                print("it is not cln")
                train_embeddings1 = train_embeddings1.detach().numpy()
                test_embeddings1 = test_embeddings1.detach().numpy()
              if(flag2==0):
                print("it is not cln2")
                train_embeddings2 = train_embeddings2.detach().numpy()
                test_embeddings2 = test_embeddings2.detach().numpy()
              if(flag3==0):
                print("it is not cln3")
                train_embeddings3 = train_embeddings3.detach().numpy()
                test_embeddings3 = test_embeddings3.detach().numpy()
              if(flag4==0):
                print("it is not cln3")
                train_embeddings4 = train_embeddings4.detach().numpy()
                test_embeddings4 = test_embeddings4.detach().numpy()
              if(flag5==0):
                print("it is not cln3")
                train_embeddings5 = train_embeddings5.detach().numpy()
                test_embeddings5 = test_embeddings5.detach().numpy()
              if(flag6==0):
                print("it is not cln3")
                train_embeddings6 = train_embeddings6.detach().numpy()
                test_embeddings6 = test_embeddings6.detach().numpy()


              if(Fold==1):
                labels=labels.unsqueeze(1)
              #   print("shape of label",labels.shape)
              #=====with attention=========================================
              accuracy,sensitivity, precision,f1,specificity,auc_value=classifyNodes1(train_idx,test_idx,train_embeddings1,train_embeddings2,train_embeddings3,train_embeddings4,train_embeddings5, train_embeddings6,\
                                                                test_embeddings1,test_embeddings2 ,test_embeddings3,test_embeddings4,test_embeddings5,test_embeddings6,labels)
             

              acc=acc+accuracy
              sen=sen+sensitivity
              pre=pre+precision
              f1_val=f1_val+f1
              spe=spe+specificity
              auc=auc+auc_value
              acc_l.append(accuracy)
              sen_l.append(sensitivity)
              pre_l.append(precision)
              f1_l.append(f1)
              spe_l.append(specificity)
              auc_l.append(auc_value)

    acc=acc/10
    sen=sen/10
    pre=pre/10
    f1_val=f1_val/10
    spe=spe/10
    auc=auc/10

    acc, std_dev_acc = np.mean(acc_l), np.std(acc_l)
    sen, std_dev_sen = np.mean(sen_l), np.std(sen_l)
    pre, std_dev_pre = np.mean(pre_l), np.std(pre_l)
    f1_val, std_dev_f1 = np.mean(f1_l), np.std(f1_l)
    spe, std_dev_spe = np.mean(spe_l), np.std(spe_l)
    auc, std_dev_auc = np.mean(auc_l), np.std(auc_l)

    rounded_accuracy = round(acc, 4)
    file1.write("Acc:,")
    file1.write(str(rounded_accuracy))
    file1.write("\n")

    rounded_sen = round(sen, 4)
    file1.write("sen:,")
    file1.write(str(rounded_sen))
    file1.write("\n")

    rounded_pre = round(pre, 4)
    file1.write("pre:,")
    file1.write(str(rounded_pre))
    file1.write("\n")

    rounded_f1_val = round(f1_val, 4)
    file1.write("f1_val:,")
    file1.write(str(rounded_f1_val))
    file1.write("\n")

    rounded_auc = round(auc, 4)
    file1.write("Auc:,")
    file1.write(str(rounded_auc))
    file1.write("\n")

    print(f"Accuracy: {acc:.4f}", "sttdv",std_dev_acc)
    print(f"Sensitivity: {sen:.4f}","sttdv",std_dev_sen)
    print(f"Precision: {pre:.4f}","sttdv",std_dev_pre)
    print(f"F1 Score: {f1_val:.4f}","sttdv",std_dev_f1 )
    print(f"specificity: {spe:.4f}","sttdv",std_dev_spe)
    print(f"auc: {auc:.4f}","sttdv",std_dev_auc)

    all_avg_acc=all_avg_acc+acc
    all_avg_sen=all_avg_sen+sen
    all_avg_pre=all_avg_pre+pre
    all_avg_f1_val=all_avg_f1_val+f1_val
    all_avg_spe=all_avg_spe+spe
    all_avg_auc=all_avg_auc+auc


all_avg_acc=all_avg_acc/1
all_avg_acc=round(all_avg_acc,4)
all_avg_sen=all_avg_sen/1
all_avg_sen=round(all_avg_sen,4)
all_avg_pre=all_avg_pre/1
all_avg_pre=round(all_avg_pre,4)
all_avg_f1_val=all_avg_f1_val/1
all_avg_f1_val=round(all_avg_f1_val,4)
all_avg_auc=all_avg_auc/1
all_avg_auc=round(all_avg_auc,4)
file1.write("final result:,")
file1.write(str(all_avg_acc))
file1.write(str(all_avg_sen))
file1.write(str(all_avg_pre))
file1.write(str(all_avg_f1_val))


file1.close()
print(" no attention")
print(f"Accuracy: {all_avg_acc:.4f}")
print(f"Sensitivity: {all_avg_sen:.4f}")
print(f"Precision: {all_avg_pre:.4f}")
print(f"F1 Score: {all_avg_f1_val:.4f}")
print(f"auc Score: {all_avg_auc:.4f}")

