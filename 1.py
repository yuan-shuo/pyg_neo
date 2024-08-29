import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score
from neo4j import GraphDatabase

# 连接到Neo4j数据库
class Neo4jGraph:
    def __init__(self, uri, user, pwd, db_name):
        self._driver = GraphDatabase.driver(uri, auth=(user, pwd), database=db_name)

    def close(self):
        self._driver.close()

    def query(self, query):
        with self._driver.session() as session:
            result = session.run(query)
            return [record for record in result]

# 从Neo4j提取图数据并创建Data对象
def extract_graph_data_from_neo4j(neo4j_graph, query):
    result = neo4j_graph.query(query)
    # 假设结果是一个列表，每个元素是一个字典，包含节点和边的信息
    # 这里需要根据您的具体数据结构进行调整
    nodes = {node['id']: node for node in result}  # 节点字典
    edges = [(src['id'], dst['id']) for src, dst in zip(result[::2], result[1::2])]  # 边列表

    # 创建特征矩阵x和边索引edge_index
    x = torch.tensor([node['features'] for node in nodes.values()], dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # 创建Data对象
    data = Data(x=x, edge_index=edge_index)
    return data

# 假设已经有了图数据对象`data`
uri = "bolt://localhost:7687"
user = "neo4j"
pwd = "1742359208ys"
db_name = "neo4j"

# 创建Neo4jGraph实例
neo4j_graph = Neo4jGraph(uri, user, pwd, db_name)

# 定义Cypher查询
query = """
MATCH (n) RETURN n AS node
"""

# 提取图数据并创建Data对象
data = extract_graph_data_from_neo4j(neo4j_graph, query)

# 此处添加GNN模型定义和训练的代码

# 关闭Neo4j连接
neo4j_graph.close()

# 定义GNN模型
class GNN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# 初始化模型、优化器和损失函数
model = GNN(data.num_node_features, data.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# 训练模型
model.train()
for data in data_loader:  # 假设有一个DataLoader
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# 特征提取
model.eval()
gnn_features = model(data.x, data.edge_index)

# 将特征转换为Pandas DataFrame
df_features = pd.DataFrame(gnn_features.numpy())

# 定义目标变量
target = pd.Series(data.y.numpy())

# 训练随机森林模型
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(df_features, target)

# 预测和评估模型（略）