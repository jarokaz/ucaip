#!/bin/bash
set -e

# Fill in *all* the following env vars for your project
PROJECT_NAME=
TB_NAME=
GCS_OUTPUT_PATH=
SA_EMAIL=
TRAINING_CONTAINER=
##### -- mostly no changes required for the rest of the file


ENDPOINT=us-central1-aiplatform.googleapis.com
TB_WEB_APP=https://tensorboard-gcp-prod.uc.r.appspot.com/experiment

INVOCATION_TIMESTAMP=$(date  +'%Y%m%d_%H%M%S')
JOB_NAME=tbgcp_job_${INVOCATION_TIMESTAMP}
BASE_OUTPUT_DIR=$GCS_OUTPUT_PATH/$JOB_NAME

JOB_RES=$(curl \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
-H "Content-Type: application/json" \
-X POST \
-d "{
'displayName':'$JOB_NAME',
'jobSpec':{
'workerPoolSpecs':[
     {
        'replicaCount': '1',
        'machineSpec': {
          'machineType': 'n1-standard-8',
        },
        'containerSpec': {
          'imageUri': '$TRAINING_CONTAINER',
        }
      }
],
'base_output_directory': {
'output_uri_prefix': '$BASE_OUTPUT_DIR',
 },
'serviceAccount': '$SA_EMAIL',
'tensorboard':'$TB_NAME',
}
}" \
https://${ENDPOINT}/v1beta1/projects/${PROJECT_NAME}/locations/us-central1/customJobs)
echo
echo "CustomJob created"
echo $JOB_RES
JOB_ID=$(echo $JOB_RES| head -n4 | tail -n1 | cut -d'/' -f6 | cut -d'"' -f1)
TB_URL=$(echo $TB_WEB_APP/$(echo $TB_NAME|sed "s/\//\+/g")+experiments+$JOB_ID)
echo
echo "Tensorboard experiment URL (training job can take up to 5 mins to start.)"
echo $TB_URL
