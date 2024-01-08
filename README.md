### Install dependencies
```shell
make install
```
### Start server in debug mode
```shell
python -m flask run --debug
```
### Call predict API
```shell
curl -vX POST localhost:5000/api/v1/predict \
 --header 'Content-Type: application/json' \
 --data '[
    164.36000061035156,
    166.50999450683594,
    166.47000122070312,
    167.64999389648438
]'
```
