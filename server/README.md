### Install pipenv
```shell
make setup
```
### Install dependencies
```shell
make i
```
### Start server in debug mode
```shell
make run
```
### Define variables
```shell
#fin_predictor_host=127.0.0.1
# or
fin_predictor_host="$(minikube ip)"
```
### Health check
```shell
curl -v $fin_predictor_host:5000/healthz/live
curl -v $fin_predictor_host:5000/healthz/ready
```
### Call predict API
```shell
curl -v "http://$fin_predictor_host:5000/api/v1/predict" \
 --header 'Content-Type: application/json' \
 --data '{
  "prices": [
    164.36000061035156,
    166.50999450683594,
    166.47000122070312,
    167.64999389648438
  ]
 }'
```
### Docker commands
#### Build an image
```shell
docker build -t finance-predictor:latest .
```
#### Run tensorflow in interactive mode
```shell
docker run -it --rm --name tensorflow tensorflow/tensorflow:2.15.0 /bin/bash
```
#### Run finance-predictor in interactive mode and override entry point
```shell
docker run -it --rm --name finance-predictor \
  --entrypoint /bin/bash \
  finance-predictor:latest
```
#### Create network
```shell
docker network create finance-predictor-net
```
#### Run finance-predictor
```shell
docker run \
  --rm \
  --name finance-predictor \
  --hostname finance-predictor \
  --network finance-predictor-net \
  --publish 5000:5000 \
  finance-predictor:latest
```
#### Connect to running container 
```shell
docker exec -it finance-predictor /bin/bash
```
