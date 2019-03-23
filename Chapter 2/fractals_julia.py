
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
     

#JULIA SET
Y, X = np.mgrid[-2:2:0.005, -2:2:0.005]

#Definiamo il punto corrente 
Z = X+1j*Y
Z = tf.constant(Z.astype("complex64"))

zs = tf.Variable(Z)
ns = tf.Variable(tf.zeros_like(Z, "float32"))

#c = complex(0.0,0.75)
#c = complex(-1.5,-1.5)
sess = tf.InteractiveSession()

#tf.initialize_all_variables().run()
tf.global_variables_initializer().run()

c = complex(0.0,0.75)
# Compute the new values of z: z^2 + x
zs_ = zs*zs - c
#zs_ = zs*zs - c

# Have we diverged with this new value?
not_diverged = tf.abs(zs_) < 4

step = tf.group(
  zs.assign(zs_),
  ns.assign_add(tf.cast(not_diverged, "float32"))
  )

for i in range(200): step.run()

plt.imshow(ns.eval())
plt.show()
