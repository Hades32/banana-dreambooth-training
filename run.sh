#!/bin/bash

# maybe make helper website from this

start=$(date)

callID=$(curl -s -XPOST 'https://api.banana.dev/start/v4/' -H 'Content-Type: application/json' -d '
{
    "startOnly": true,
    "apiKey": "'$API_KEY'",
    "modelKey": "'$MODEL_KEY'",
    "modelInputs": {
        "S3_ENDPOINT": "'$S3_ENDPOINT'",
        "S3_BUCKET": "'$S3_BUCKET'",
        "S3_KEY": "'$S3_KEY'",
        "S3_SECRET": "'$S3_SECRET'",
        "S3_REGION": "'$S3_REGION'",
        "file_id": "rita"
    }
}' | jq -j .callID)

echo "started ${callID} at ${start}"

while export result=$(curl -s -XPOST 'https://api.banana.dev/check/v4/' -H 'Content-Type: application/json' -d'
{    
    "apiKey": "'$API_KEY'",    
    "callID": "'$callID'"
}') && echo "$result" | grep running > /dev/null ; do 
    echo -n .
done

echo "$result" | jq .
echo "$start until $(date)"
