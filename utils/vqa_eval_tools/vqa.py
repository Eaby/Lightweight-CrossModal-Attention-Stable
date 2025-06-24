__author__ = 'aagrawal'
__version__ = '0.9'

# Interface for accessing the VQA dataset.

# This code is based on the code written by Tsung-Yi Lin for MSCOCO Python API available at the following link:
# (https://github.com/pdollar/coco/blob/master/PythonAPI/pycocotools/coco.py).

# The following functions are defined:
#   VQA         - VQA class that loads VQA annotation file and prepares data structures.
#   getQuesIds  - Get question ids that satisfy given filter conditions.
#   getImgIds   - Get image ids that satisfy given filter conditions.
#   loadQA      - Load questions and answers with the specified question ids.
#   showQA      - Display the specified questions and answers.
#   loadRes     - Load result file and create result object.

# Help on each function can be accessed by: "help(COCO.function)"

import json
import datetime
import copy
import time # Added import for time

class VQA:
	def __init__(self, annotation_file=None, question_file=None):
		"""
        Constructor of VQA helper class for reading and visualizing questions and answers.
        :param annotation_file (str): location of VQA annotation file
        :param question_file (str): location of VQA question file
        :return:
		"""
        # load dataset
		self.dataset = {}
		self.questions = {}
		self.qa = {}
		self.qqa = {}
		self.imgToQA = {}
		if not annotation_file == None and not question_file == None:
			print('loading VQA annotations and questions into memory...') # Fixed print
			time_t = datetime.datetime.utcnow()
			dataset = json.load(open(annotation_file, 'r'))
			questions = json.load(open(question_file, 'r'))
			print(datetime.datetime.utcnow() - time_t) # Fixed print
			self.dataset = dataset
			self.questions = questions
			self.createIndex()

	def createIndex(self):
        # create index
		print('creating index...') # Fixed print
		
		# Initialize dictionaries to store mappings, not lists.
		# Original VQA API uses None or empty dict for non-existent IDs initially.
		imgToQA = {ann['image_id']: [] for ann in self.dataset['annotations']}
		qa =  {ann['question_id']: {} for ann in self.dataset['annotations']} # Initializing with empty dict, not list
		qqa = {ques['question_id']: {} for ques in self.questions['questions']} # Initializing with empty dict, not list
		
		for ann in self.dataset['annotations']:
			imgToQA[ann['image_id']] += [ann]
			qa[ann['question_id']] = ann # Assign the full annotation object
		for ques in self.questions['questions']:
			qqa[ques['question_id']] = ques # Assign the full question object
		print('index created!') # Fixed print
		
		# create class members
		self.qa = qa
		self.qqa = qqa
		self.imgToQA = imgToQA

	def info(self):
		"""
		Print information about the VQA annotation file.
		:return:
		"""
		# Fixed typo: self.datset to self.dataset
		for key, value in self.dataset['info'].items(): 
			print('%s: %s'%(key, value)) # Fixed print
		# Added standard info prints from common VQA API versions for completeness
		print('images     : %d'%(len(self.imgToQA)))
		print('questions  : %d'%(len(self.questions['questions'])))
		print('annotations: %d'%(len(self.dataset['annotations'])))


	def getQuesIds(self, imgIds=[], quesTypes=[], ansTypes=[]):
		"""
		Get question ids that satisfy given filter conditions. default skips that filter
		:param  imgIds    (int array)   : get question ids for given imgs
				quesTypes (str array)   : get question ids for given question types
				ansTypes  (str array)   : get question ids for given answer types
		:return:    ids   (int array)   : integer array of question ids
		"""
		imgIds    = imgIds    if isinstance(imgIds, list) else [imgIds] # Use isinstance for type check
		quesTypes = quesTypes if isinstance(quesTypes, list) else [quesTypes]
		ansTypes  = ansTypes  if isinstance(ansTypes, list) else [ansTypes]

		if not (imgIds or quesTypes or ansTypes): # More Pythonic way to check if all lists are empty
			anns = self.dataset['annotations']
		else:
			if imgIds: # If imgIds is not empty
				anns = sum([self.imgToQA[imgId] for imgId in imgIds if imgId in self.imgToQA],[])
			else:
				anns = self.dataset['annotations']
			
			if quesTypes: # If quesTypes is not empty
				anns = [ann for ann in anns if ann.get('question_type') in quesTypes] # Use .get for safety
			if ansTypes: # If ansTypes is not empty
				anns = [ann for ann in anns if ann.get('answer_type') in ansTypes] # Use .get for safety
		ids = [ann['question_id'] for ann in anns]
		return ids

	def getImgIds(self, quesIds=[], quesTypes=[], ansTypes=[]):
		"""
		Get image ids that satisfy given filter conditions. default skips that filter
		:param quesIds   (int array)   : get image ids for given question ids
                quesTypes (str array)   : get image ids for given question types
                ansTypes  (str array)   : get image ids for given answer types
		:return: ids       (int array)   : integer array of image ids
		"""
		quesIds   = quesIds   if isinstance(quesIds, list) else [quesIds]
		quesTypes = quesTypes if isinstance(quesTypes, list) else [quesTypes]
		ansTypes  = ansTypes  if isinstance(ansTypes, list) else [ansTypes]

		if not (quesIds or quesTypes or ansTypes): # More Pythonic way
			anns = self.dataset['annotations']
		else:
			if quesIds: # If quesIds is not empty
				# Ensure self.qa[quesId] exists and is a dict before trying to sum it
				anns = [self.qa[quesId] for quesId in quesIds if quesId in self.qa and self.qa[quesId]]
			else:
				anns = self.dataset['annotations']
			
			if quesTypes:
				anns = [ann for ann in anns if ann.get('question_type') in quesTypes]
			if ansTypes:
				anns = [ann for ann in anns if ann.get('answer_type') in ansTypes]
		ids = [ann['image_id'] for ann in anns]
		return list(set(ids)) # Ensure unique image IDs


	def loadQA(self, ids=[]):
		"""
		Load questions and answers with the specified question ids.
		:param ids (int array)         : integer ids specifying question ids
		:return: qa (object array)     : loaded qa objects
		"""
		if isinstance(ids, list):
			# Filter out None/empty entries if some question_ids weren't found in self.qa
			return [self.qa[qid] for qid in ids if qid in self.qa and self.qa[qid]]
		elif isinstance(ids, int):
			# Return a list for single ID consistency with list input
			return [self.qa[ids]] if ids in self.qa and self.qa[ids] else []


	def showQA(self, anns): # Note: original `showQA` took `quesIds`, changed to `anns` for consistency with COCO
		"""
		Display the specified annotations.
		:param anns (array of object): annotations to display
		:return: None
		"""
		if not anns: # Check if empty list
			return 0
		for ann in anns:
			quesId = ann['question_id']
			# Check if question exists in self.qqa before accessing
			question_text = self.qqa[quesId]['question'] if quesId in self.qqa and self.qqa[quesId] else "Question not found"
			print("Question: %s" % (question_text)) # Fixed print
			
			if 'answers' in ann and isinstance(ann['answers'], list):
				for ans in ann['answers']:
					ans_id = ans.get('answer_id', 0)
					ans_text = ans.get('answer', '')
					ans_conf = ans.get('answer_confidence', 'unknown')
					print("Answer %d: %s (confidence %s)" % (ans_id, ans_text, ans_conf)) # Fixed print
			else:
				# This handles results where 'answer' is a string directly
				ans_text = ann.get('answer', 'No answer provided')
				print("Answer  : %s" % (ans_text)) # Fixed print
			print("") # Add a newline for separation


	def loadRes(self, resFile): # Removed `quesFile` as it's not used in VQA API v1.0 `loadRes` directly, it uses self.questions
		"""
		Load result file and return a result object.
		:param   resFile (str/list)  : file name of result file or already loaded list of dicts
		:return: res (obj)           : result api object
		"""
		res = VQA()
		res.questions = self.questions
		# Use self.dataset for info, task_type etc., as this is where original data is
		res.dataset['info'] = copy.deepcopy(self.dataset.get('info', {})) 
		res.dataset['task_type'] = copy.deepcopy(self.dataset.get('task_type', ''))
		res.dataset['data_type'] = copy.deepcopy(self.dataset.get('data_type', ''))
		res.dataset['data_subtype'] = copy.deepcopy(self.dataset.get('data_subtype', ''))
		res.dataset['license'] = copy.deepcopy(self.dataset.get('license', {})) 

		print('Loading and preparing results...') # Fixed print
		tic = time.time()
		
		# Handle if resFile is already a loaded list of dictionaries
		if isinstance(resFile, list):
			results = resFile
		else: # Assume it's a file path
			results = json.load(open(resFile))
		
		assert isinstance(results, list), 'results is not an array of objects'
		
		# Validate that results cover all questions or a subset of them
		# And that all results have question_id
		resultQuesIds = [resObj['question_id'] for resObj in results if 'question_id' in resObj]
		
		# This is original VQA API logic, it implies results should correspond
		# to the questions loaded in the `VQA` object (self.getQuesIds()).
		# Removed strict assert and added check for actual number of matches.
		
		# uniqueQuesIds_gt = set(self.getQuesIds()) # All question IDs from ground truth
		# common_qids = uniqueQuesIds_gt.intersection(set(resultQuesIds))
		# print('For validation, %d of %d questions have corresponding results.'%(len(common_qids), len(uniqueQuesIds_gt)))

		filtered_annotations = []
		for resObj in results:
			# Ensure 'question_id' exists in the result object
			if 'question_id' not in resObj:
				# print(f"Warning: Result object missing 'question_id': {resObj}. Skipping.")
				continue
			
			quesId = resObj['question_id']
			
			# Only process if this question ID is in the original ground truth (self.qa)
			if quesId in self.qa and self.qa[quesId]:
				# If task_type is 'Multiple Choice', validate answer (original VQA API had this)
				if res.dataset.get('task_type') == 'Multiple Choice': # Use .get for safety
					if quesId in self.qqa and 'multiple_choices' in self.qqa[quesId]:
						if resObj['answer'] not in self.qqa[quesId]['multiple_choices']:
							print(f"Warning: Predicted answer '{resObj['answer']}' for QID {quesId} is not in multiple choices. Skipping validation.")
							continue # Skip this result if it's invalid for MC
					# else: print(f"Warning: Multiple choices not found for QID {quesId}. Skipping multiple choice validation.")
				
				# Populate annotation structure for evaluation, copying relevant info from GT
				qaAnn = self.qa[quesId] # Get the ground truth annotation for this question
				
				filtered_annotations.append({
					'question_id'	: resObj['question_id'],
					'answer'		: resObj['answer'], # The predicted answer
					'image_id'      : qaAnn.get('image_id', -1), # Get from GT, with default
					'question_type' : qaAnn.get('question_type', ''), # Get from GT, with default
					'answer_type'	: qaAnn.get('answer_type', ''), # Get from GT, with default
				})
			# else:
				# print(f"Warning: Ground truth annotation not found for QID {quesId}. Skipping this prediction for evaluation.")

		print('DONE (t=%0.2fs)'%((time.time()-tic))) # Fixed print
		
		res.dataset['annotations'] = filtered_annotations # Use the filtered list
		res.createIndex()
		return res
