DIR=$(dirname "$(readlink -f "$0")")

bash $DIR/data.sh
bash $DIR/train.sh $1
bash $DIR/val.sh
bash $DIR/test.sh
