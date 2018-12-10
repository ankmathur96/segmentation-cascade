source config.sh
echo $ZONE
gcloud compute instances create $INSTANCE_NAME --project=$PROJECT --boot-disk-size=200GB --zone=$ZONE --image-family=$IMAGE_FAMILY --image-project=deeplearning-platform-release --maintenance-policy=TERMINATE --accelerator=$INSTANCE_SPEC --metadata="install-nvidia-driver=True"
