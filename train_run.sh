if [[ "$#" -gt 2 ||  "$#" -lt 2 ]]; then
  echo "Usage: $0 DATASET_NAME DATA_DIR"
  exit
fi

TRAINING_DATA_DIR=$1
VALIDATION_DATA_DIR=$2

echo 'env settings ... '

pip install -r requirements.txt



if [ $? -eq 0 ];then

  if [$1 -eq "pcb"]; then

    echo 'download yolov3 weights ... '
    mkdir data
    wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights
    python convert.py --weights ./data/yolov3.weights --output ./checkpoints/yolov3.tf

    echo 'download pcb dataset'
    mkdir pcb_dataset
    gdown https://drive.google.com/uc?id=1qn0mLFV7NBbmw6-ZTuF_tN_oJxRHx-XR -O ./pcb_dataset/pcb_dataset.tfrecord
    gdown https://drive.google.com/uc?id=1ReribxK5PHjC6cKPVMTRX1dIP3qIt7PF -O ./pcb_dataset/pcb.names

    echo 'start train ... '
    python train.py \
        --names pcb \
        --batch_size 8 \
        --dataset ./pcb_dataset/pcb_dataset.tfrecord \
        --epochs 2 --classes ./pcb_dataset/pcb.names \
        --num_classes 6 \
        --mode eager_tf \
        --transfer darknet \
        --weights ./checkpoints/yolov3.tf \
        --output ./pcb_model \
        --weights_num_classes 80

  elif [$1 -eq "brain"]; then

    echo 'download pretrained weights ... '
    
    echo 'start train ... '
    python train.py \
        --names 
        --batch_size 8 \
        --dataset ./pcb_dataset/pcb_dataset.tfrecord \
        --epochs 2 --classes ./pcb_dataset/pcb.names \
        --num_classes 6 \
        --mode eager_tf \
        --transfer darknet \
        --weights ./checkpoints/yolov3.tf \
        --output ./pcb_model \
        --weights_num_classes 80

  else
    echo "please input dataset name [pcb, brain]"
    exit 9

else
  echo "install failed"
  exit 9
fi


echo "finished ..."