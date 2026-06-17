"""Construção dos modelos (6-classes e TIPO 3-classes), loss com logit-adjustment e
ensemble. Mantém a arquitetura do Leo (BiLSTM + MultiHeadAttention), mas:
  - mascara o padding de verdade (pooling só nos frames reais) — antes o padding
    padronizado virava lixo que o BiLSTM comia;
  - o Dense final sai em LOGITS (sem softmax) para o logit-adjustment funcionar;
    a inferência aplica softmax à parte (predict_proba) — treino == inferência.
"""
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Bidirectional, LSTM, Dense, Dropout, MultiHeadAttention,
                                      Input, Add, LayerNormalization, Layer)
from tensorflow.keras.regularizers import l2


@keras.saving.register_keras_serializable(package="boxe")
class FrameMask(Layer):
    """Máscara por frame a partir da entrada CRUA: 1 onde algum canal != 0 (frame real),
    0 no padding/frame todo-zero. Calculada da entrada, não propagada — assim o LSTM roda
    sem Masking (kernel cuDNN, rápido) e a máscara é usada só no pooling."""
    def call(self, x):
        return tf.cast(tf.reduce_any(tf.not_equal(x, 0.0), axis=-1), tf.float32)  # (B,T)


@keras.saving.register_keras_serializable(package="boxe")
class MaskedMean(Layer):
    """Média temporal só dos frames reais. inputs = [x (B,T,C), mask (B,T)]. Ignora
    padding E frames internos todo-zero (detecção perdida) — robusto a máscara não
    right-padded (que o cuDNN LSTM não toleraria, por isso o mascaramento fica aqui)."""
    def call(self, inputs):
        x, mask = inputs
        m = tf.cast(mask, x.dtype)[..., None]               # (B,T,1)
        s = tf.reduce_sum(x * m, axis=1)
        d = tf.reduce_sum(m, axis=1)
        return s / tf.maximum(d, tf.constant(1e-6, x.dtype))


def build_model(input_shape, n_classes):
    """BiLSTM(64) + MultiHeadAttention + pooling mascarado -> LOGITS (n_classes).
    Sem camada Masking: o LSTM roda os 25 frames (kernel cuDNN, rápido, igual ao original
    do Leo); o padding é descartado só no pooling (MaskedMean). input_shape = (25, n_feat)."""
    i = Input(shape=input_shape)
    mask = FrameMask()(i)                                   # (B,25) frames reais
    x = Bidirectional(LSTM(64, return_sequences=True))(i)   # cuDNN (sem dropout interno)
    x = Dropout(0.3)(x)
    a = MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
    x = Add()([x, a])
    x = LayerNormalization()(x)
    x = MaskedMean()([x, mask])                             # pooling só dos frames reais
    x = Dense(64, activation="relu", kernel_regularizer=l2(0.0005))(x)
    x = Dropout(0.3)(x)
    return Model(i, Dense(n_classes)(x))                    # LOGITS (sem softmax)


@keras.saving.register_keras_serializable(package="boxe")
class LogitAdjustedLoss(tf.keras.losses.Loss):
    """Balanced softmax / logit-adjustment (Menon et al. ICLR 2021): soma tau*log(pi)
    aos logits no TREINO (pi = freq de classe) -> correção Bayes-ótima do prior, sobe o
    recall das classes raras (Rear Hook). A INFERÊNCIA usa os logits crus (sem ajuste)."""
    def __init__(self, log_pi, tau=1.0, label_smoothing=0.05, name="logit_adjusted", **kwargs):
        super().__init__(name=name, **kwargs)
        self.log_pi = tf.constant(np.asarray(log_pi, np.float32))
        self.tau = float(tau)
        self.label_smoothing = float(label_smoothing)
        self._cce = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=self.label_smoothing)

    def call(self, y_true, y_pred):
        return self._cce(y_true, y_pred + self.tau * self.log_pi)

    def get_config(self):
        c = super().get_config()
        c.update(log_pi=self.log_pi.numpy().tolist(), tau=self.tau,
                 label_smoothing=self.label_smoothing)
        return c


def compile_model(model, log_pi, tau=1.0, label_smoothing=0.05, lr=5e-4):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=LogitAdjustedLoss(log_pi, tau, label_smoothing),
                  metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")])
    return model


def predict_proba(model, X, batch_size=512):
    """Softmax aplicada FORA do modelo (que sai em logits). Fonte única de probabilidade
    para treino e inferência."""
    logits = model.predict(X, batch_size=batch_size, verbose=0)
    return tf.nn.softmax(logits, axis=-1).numpy()


def predict_proba_ensemble(models, X, batch_size=512):
    """Média das softmaxes de um ensemble (calibração robusta sob shift + stream mais suave)."""
    probs = [predict_proba(m, X, batch_size) for m in models]
    return np.mean(probs, axis=0)


def log_prior(y_ids, n_classes):
    """log(pi) das frequências de classe no treino, para o logit-adjustment."""
    counts = np.bincount(np.asarray(y_ids), minlength=n_classes).astype(np.float32)
    pi = counts / counts.sum()
    return np.log(np.clip(pi, 1e-8, None)).astype(np.float32)
