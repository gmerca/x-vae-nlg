
# Prepare the data for iVAE model.

cd ../../

python iVAE/code/preprocess.py \
--trainfile=data/worldtree/train.txt \
--valfile=data/worldtree/val.txt \
--testfile=data/worldtree/test.txt \
--outputfile=data/worldtree/wt
