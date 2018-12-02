from django.test import TestCase

# Create your tests here.

import sys, os

path = os.path.join(os.path.dirname(os.path.dirname(__file__)) + '/model/tfidf')
print(path)