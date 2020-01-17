from flask import render_template, jsonify, Flask, redirect, url_for, request
from app import app
import random
import os
# import tensorflow as tf
# import numpy as np
# import sys
# import spacy

# nlp = spacy.load('en')

# sys.path.insert(0, "/content/bert_experimental")

# from bert_experimental.finetuning.text_preprocessing import build_preprocessor
# from bert_experimental.finetuning.graph_ops import load_graph
# restored_graph = load_graph("models/frozen_graph.pb")
# graph_ops = restored_graph.get_operations()
# input_op, output_op = graph_ops[0].name, graph_ops[-1].name
# x = restored_graph.get_tensor_by_name(input_op + ':0')
# y = restored_graph.get_tensor_by_name(output_op + ':0')
# preprocessor = build_preprocessor("./uncased_L-12_H-768_A-12/vocab.txt", 256)
# py_func = tf.numpy_function(preprocessor, [x], [tf.int32, tf.int32, tf.int32], name='preprocessor')
# py_func = tf.numpy_function(preprocessor, [x], [tf.int32, tf.int32, tf.int32])
# sess = tf.Session(graph=restored_graph)
# delimiter = " ||| "

@app.route('/')
def index1():
    return render_template('index.html', title='Home')

	
@app.route('/predict', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      exp_st = request.form.get('exp')
      job_st = request.form.get('job')
      # y_out = sess.run(y, feed_dict={
      #   x: pd.DataFrame([delimiter.join((exp_st, job_st ))], columns=['name'])
      # })
      # doc1 = nlp(exp_st)
      # doc2 = nlp(job_st )
      # y_out2 = doc1.similarity(doc2)
   return render_template('index.html', title='Success', predictions=80, predictions_sp =75, exp=exp_st, job= job_st)


@app.route('/index')
def index():
    return render_template('index.html', title='Home')

@app.route('/map')
def map():
    return render_template('map.html', title='Map')


@app.route('/map/refresh', methods=['POST'])
def map_refresh():
    points = [(random.uniform(48.8434100, 48.8634100),
               random.uniform(2.3388000, 2.3588000))
              for _ in range(random.randint(2, 9))]
    return jsonify({'points': points})


@app.route('/contact')
def contact():
    return render_template('contact.html', title='Contact')