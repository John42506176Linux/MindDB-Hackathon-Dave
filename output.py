from diagrams import Diagram, Cluster, Edge
from diagrams.aws.compute import EC2
from diagrams.aws.network import ELB
from diagrams.aws.database import RDS
from diagrams.onprem.client import Users
from diagrams.onprem.network import Internet
from diagrams.generic.blank import Blank

with Diagram("Three-Tier Architecture", filename="diagram.png"):
    users = Users("Users")
    internet = Internet("Internet")
    
    with Cluster("AWS Cloud"):
        with Cluster("Presentation Tier"):
            load_balancer = ELB("Load Balancer")
        
        with Cluster("Logic Tier"):
            app_servers = [EC2("App Server 1"),
                           EC2("App Server 2"),
                           EC2("App Server 3")]
        
        with Cluster("Data Tier"):
            database = RDS("Database")
    
    users >> Edge(label="HTTPS") >> internet >> Edge(label="HTTPS") >> load_balancer
    load_balancer >> Edge(label="HTTPS") >> app_servers
    app_servers >> Edge(label="JDBC") >> database