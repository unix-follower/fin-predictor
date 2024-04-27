#### Create namespace
```shell
kubectl create namespace fin-processor
```
#### Delete namespace
```shell
kubectl delete namespace fin-processor
```
### Install chart
```shell
helm install fin-predictor ./fin-predictor \
  -n fin-processor \
  --values ./fin-predictor/values.yaml
```
### Show manifest
```shell
helm get manifest -n fin-processor fin-predictor
```
### Verify installation
```shell
kubectl get all -n fin-processor
```
### Get pod logs
```shell
kubectl logs -n fin-processor fin-predictor-<ID>
```
### Describe pod
```shell
kubectl describe -n fin-processor pod/fin-predictor-<ID>
```
### Get pod events
```shell
kubectl events -n fin-processor fin-predictor-<ID>
```
### Verify external access
```shell
nc -vz $(minikube ip) 5000
```
### Stop finance predictor
```shell
kubectl -n fin-processor scale deployment fin-predictor --replicas 0
```
### Uninstall chart
```shell
helm uninstall -n fin-processor fin-predictor
```
