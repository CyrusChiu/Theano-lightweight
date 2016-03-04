# Written by https://github.com/uoguelph-mlrg/theano_alexnet
# CyrusChiu, Twbadkid modified
import numpy as np
import os

def save_weights(layers, weights_dir, epoch):
    for idx in range(len(layers)):
        if hasattr(layers[idx], 'W'):
            layers[idx].W.save_weight(
                weights_dir, 'W' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'W0'):
            layers[idx].W0.save_weight(
                weights_dir, 'W0' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'W1'):
            layers[idx].W1.save_weight(
                weights_dir, 'W1' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'b'):
            layers[idx].b.save_weight(
                weights_dir, 'b' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'b0'):
            layers[idx].b0.save_weight(
                weights_dir, 'b0' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'b1'):
            layers[idx].b1.save_weight(
                weights_dir, 'b1' + '_' + str(idx) + '_' + str(epoch))
        # RNN
        if hasattr(layers[idx], 'Wh'):
            layers[idx].Wh.save_weight(
                weights_dir, 'Wh' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'Wi'):
            layers[idx].Wi.save_weight(
                weights_dir, 'Wi' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'Wo'):
            layers[idx].Wo.save_weight(
                weights_dir, 'Wo' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'h0'):
            layers[idx].h0.save_weight(
                weights_dir, 'h0' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'bh'):
            layers[idx].bh.save_weight(
                weights_dir, 'bh' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'by'):
            layers[idx].by.save_weight(
                weights_dir, 'by' + '_' + str(idx) + '_' + str(epoch))
        # LSTM
        if hasattr(layers[idx], 'Ui'):
            layers[idx].Ui.save_weight(
                weights_dir, 'Ui' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'bi'):
            layers[idx].bi.save_weight(
                weights_dir, 'bi' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'Wc'):
            layers[idx].Wc.save_weight(
                weights_dir, 'Wc' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'Uc'):
            layers[idx].Uc.save_weight(
                weights_dir, 'Uc' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'bc'):
            layers[idx].bc.save_weight(
                weights_dir, 'bc' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'Wf'):
            layers[idx].Wf.save_weight(
                weights_dir, 'Wf' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'Uf'):
            layers[idx].Uf.save_weight(
                weights_dir, 'Uf' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'bf'):
            layers[idx].bf.save_weight(
                weights_dir, 'bf' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'Uo'):
            layers[idx].Uo.save_weight(
                weights_dir, 'Uo' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'bo'):
            layers[idx].bo.save_weight(
                weights_dir, 'bo' + '_' + str(idx) + '_' + str(epoch))
        #BLSTM
        if hasattr(layers[idx], 'Wbi'):
            layers[idx].Wbi.save_weight(
                weights_dir, 'Wbi' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'Ubi'):
            layers[idx].Ubi.save_weight(
                weights_dir, 'Ubi' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'bbi'):
            layers[idx].bbi.save_weight(
                weights_dir, 'bbi' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'Wbc'):
            layers[idx].Wbc.save_weight(
                weights_dir, 'Wbc' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'Ubc'):
            layers[idx].Ubc.save_weight(
                weights_dir, 'Ubc' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'bbc'):
            layers[idx].bbc.save_weight(
                weights_dir, 'bbc' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'Wbf'):
            layers[idx].Wbf.save_weight(
                weights_dir, 'Wbf' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'Ubf'):
            layers[idx].Ubf.save_weight(
                weights_dir, 'Ubf' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'bbf'):
            layers[idx].bbf.save_weight(
                weights_dir, 'bbf' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'Wbo'):
            layers[idx].Wbo.save_weight(
                weights_dir, 'Wbo' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'Ubo'):
            layers[idx].Ubo.save_weight(
                weights_dir, 'Ubo' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'bbo'):
            layers[idx].bbo.save_weight(
                weights_dir, 'bbo' + '_' + str(idx) + '_' + str(epoch))


def load_weights(layers, weights_dir, epoch):
    for idx in range(len(layers)):
        if hasattr(layers[idx], 'W'):
            layers[idx].W.load_weight(
                weights_dir, 'W' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'W0'):
            layers[idx].W0.load_weight(
                weights_dir, 'W0' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'W1'):
            layers[idx].W1.load_weight(
                weights_dir, 'W1' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'b'):
            layers[idx].b.load_weight(
                weights_dir, 'b' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'b0'):
            layers[idx].b0.load_weight(
                weights_dir, 'b0' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'b1'):
            layers[idx].b1.load_weight(
                weights_dir, 'b1' + '_' + str(idx) + '_' + str(epoch))
        # RNN
        if hasattr(layers[idx], 'Wh'):
            layers[idx].Wh.load_weight(
                weights_dir, 'Wh' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'Wi'):
            layers[idx].Wi.load_weight(
                weights_dir, 'Wi' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'Wo'):
            layers[idx].Wo.load_weight(
                weights_dir, 'Wo' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'h0'):
            layers[idx].h0.load_weight(
                weights_dir, 'h0' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'bh'):
            layers[idx].bh.load_weight(
                weights_dir, 'bh' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'by'):
            layers[idx].by.load_weight(
                weights_dir, 'by' + '_' + str(idx) + '_' + str(epoch))
        # LSTM
        if hasattr(layers[idx], 'Ui'):
            layers[idx].Ui.load_weight(
                weights_dir, 'Ui' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'bi'):
            layers[idx].bi.load_weight(
                weights_dir, 'bi' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'Wc'):
            layers[idx].Wc.load_weight(
                weights_dir, 'Wc' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'Uc'):
            layers[idx].Uc.load_weight(
                weights_dir, 'Uc' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'bc'):
            layers[idx].bc.load_weight(
                weights_dir, 'bc' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'Wf'):
            layers[idx].Wf.load_weight(
                weights_dir, 'Wf' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'Uf'):
            layers[idx].Uf.load_weight(
                weights_dir, 'Uf' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'bf'):
            layers[idx].bf.load_weight(
                weights_dir, 'bf' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'Uo'):
            layers[idx].Uo.load_weight(
                weights_dir, 'Uo' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'bo'):
            layers[idx].bo.load_weight(
                weights_dir, 'bo' + '_' + str(idx) + '_' + str(epoch))
        #BLSTM
        if hasattr(layers[idx], 'Wbi'):
            layers[idx].Wbi.load_weight(
                weights_dir, 'Wbi' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'Ubi'):
            layers[idx].Ubi.load_weight(
                weights_dir, 'Ubi' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'bbi'):
            layers[idx].bbi.load_weight(
                weights_dir, 'bbi' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'Wbc'):
            layers[idx].Wbc.load_weight(
                weights_dir, 'Wbc' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'Ubc'):
            layers[idx].Ubc.load_weight(
                weights_dir, 'Ubc' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'bbc'):
            layers[idx].bbc.load_weight(
                weights_dir, 'bbc' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'Wbf'):
            layers[idx].Wbf.load_weight(
                weights_dir, 'Wbf' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'Ubf'):
            layers[idx].Ubf.load_weight(
                weights_dir, 'Ubf' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'bbf'):
            layers[idx].bbf.load_weight(
                weights_dir, 'bbf' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'Wbo'):
            layers[idx].Wbo.load_weight(
                weights_dir, 'Wbo' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'Ubo'):
            layers[idx].Ubo.load_weight(
                weights_dir, 'Ubo' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'bbo'):
            layers[idx].bbo.load_weight(
                weights_dir, 'bbo' + '_' + str(idx) + '_' + str(epoch))

def dropout_load_weights(layers, weights_dir, epoch):
    for idx in range(len(layers)):
        if hasattr(layers[idx], 'W'):
            layers[idx].W.load_weight_half(
                weights_dir, 'W' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'W0'):
            layers[idx].W0.load_weight_half(
                weights_dir, 'W0' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'W1'):
            layers[idx].W1.load_weight_half(
                weights_dir, 'W1' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'b'):
            layers[idx].b.load_weight_half(
                weights_dir, 'b' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'b0'):
            layers[idx].b0.load_weight_half(
                weights_dir, 'b0' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'b1'):
            layers[idx].b1.load_weight_half(
                weights_dir, 'b1' + '_' + str(idx) + '_' + str(epoch))
