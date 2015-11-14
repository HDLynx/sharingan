__author__ = 'xavicosp'
import xml.etree.ElementTree as ET
import numpy as np

doc = ET.parse('svm_data.xml')

root = doc.getroot()
my_svm = root.find('my_svm')
support_vector = my_svm.find('support_vectors')
data = support_vector.find('_')
vctor = np.float32(data.text)
print vctor

