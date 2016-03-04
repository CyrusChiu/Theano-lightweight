from models import MLP_3layers_dropout
from models import VanillaRNN
from models import LSTM
from modelcompiler import ModelCompiler
from utils import zca_whitening

config = {'learning_rate':0.001,
          'learning_rate_decay':False,
          'n_features':, #int, feature dim
          'n_class':, #int
          'n_epochs':50,
          'batch_size':256,
          'l2':0.001,
          'weight_decay':0.0005, #only SGD available
          'momentum':0.9, #only SGD available
          'snapshot':'', #/path/to/your/snapshpt/ e.g./snapshot_81_0.704/
          'e_snapshot':''} #str, snapshot number e.g. 81

#x_train, y_train x_val, y_val = load_data()


print config
lstm = ModelCompiler(LSTM,config,optimizer="RMSprop")
lstm.train(x_train, y_train, x_val, y_val, save_model=True)
