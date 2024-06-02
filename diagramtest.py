from diagrams import Diagram, Cluster, Edge
from diagrams.aws.compute import EC2
from diagrams.aws.storage import S3
from diagrams.aws.database import RDS
from diagrams.aws.network import CloudFront
from diagrams.onprem.client import Users
from diagrams.onprem.network import Internet
from diagrams.onprem.security import Vault

with Diagram("Art Marketplace Architecture", show=False):
    users = Users("Users")
    internet = Internet("Internet")

    with Cluster("AWS Cloud"):
        cloudfront = CloudFront("CloudFront")
        ec2 = EC2("EC2 Instances")
        s3 = S3("S3 Storage")
        rds = RDS("RDS Database")
        vault = Vault("SSO & ACL")

    users >> internet >> cloudfront >> ec2
    ec2 >> s3
    ec2 >> rds
    ec2 >> vault