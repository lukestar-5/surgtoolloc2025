PROJECT_DIR=$(cd $(dirname $0); pwd)

cd $PROJECT_DIR/mmengine/
pip install -e . -v

cd $PROJECT_DIR/mmdetection/
pip install -e . -v

cd $PROJECT_DIR/mmyolo/
pip install -e . -v

pip install yapf==0.32.0
