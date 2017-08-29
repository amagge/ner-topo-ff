# ner-topo-ff
NER for toponym extraction using a feedforward deep neural network and distance supervision

Requirements:
tensorflow
numpy

To run :
python ff_model.py

Argparse prompt

usage: ff_model.py [-h] [--train TRAIN] [--test TEST] [--val VAL]
                   [--dist DIST] [--pubdir PUBDIR] [--outdir OUTDIR]
                   [--emb_loc EMB_LOC] [--embvocab EMBVOCAB]
                   [--hid_dim HID_DIM] [--lrn_rate LRN_RATE]
                   [--feat_cap FEAT_CAP] [--feat_dict FEAT_DICT]
                   [--dropout DROPOUT] [--window_size WINDOW_SIZE]
                   [--dist_epochs DIST_EPOCHS] [--train_epochs TRAIN_EPOCHS]
                   [--eval_interval EVAL_INTERVAL] [--n_classes {2,3}]
                   [--batch_size BATCH_SIZE] [--restore RESTORE] [--save SAVE]

optional arguments:
  -h, --help            show this help message and exit
  --train TRAIN         train file location
  --test TEST           test file location
  --val VAL             val file location
  --dist DIST           distance supervision files dir.
  --pubdir PUBDIR       pubmed files dir. To be production set.
  --outdir OUTDIR       Output dir for ffmodel annotated pubmed files.
  --emb_loc EMB_LOC     word2vec embedding location
  --embvocab EMBVOCAB   load top n words in word emb
  --hid_dim HID_DIM     dimension of hidden layers
  --lrn_rate LRN_RATE   learning rate
  --feat_cap FEAT_CAP   Capitalization feature
  --feat_dict FEAT_DICT
                        Dictionary feature
  --dropout DROPOUT     dropout probability
  --window_size WINDOW_SIZE
                        context window size - 3/5/7
  --dist_epochs DIST_EPOCHS
                        number of distsup epochs
  --train_epochs TRAIN_EPOCHS
                        number of train epochs
  --eval_interval EVAL_INTERVAL
                        evaluate once in _ epochs
  --n_classes {2,3}     number of classes
  --batch_size BATCH_SIZE
                        batch size of training
  --restore RESTORE     path of saved model
  --save SAVE           path to save model
