# coding=utf-8

__author__ = 'aagrawal'

import re
import sys # Added this import as it's used later by sys.stdout.write

# This code is based on the code written by Tsung-Yi Lin for MSCOCO Python API available at the following link:
# (https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/eval.py).


class VQAEval:
    def __init__(self, vqa, vqaRes, n=2):
        self.n              = n
        self.accuracy       = {}
        self.evalQA         = {}
        self.evalQuesType   = {}
        self.evalAnsType    = {}
        self.vqa            = vqa
        self.vqaRes         = vqaRes
        self.params         = {'question_id': vqa.getQuesIds()}
        self.contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
                             "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
                             "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
                             "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
                             "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
                             "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
                             "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
                             "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
                             "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
                             "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
                             "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
                             "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
                             "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
                             "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
                             "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
                             "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
                             "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
                             "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
                             "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
                             "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
                             "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
                             "youll": "you'll", "youre": "you're", "youve": "you've"}
        self.manualMap      = { 'none': '0',
                                 'zero': '0',
                                 'one': '1',
                                 'two': '2',
                                 'three': '3',
                                 'four': '4',
                                 'five': '5',
                                 'six': '6',
                                 'seven': '7',
                                 'eight': '8',
                                 'nine': '9',
                                 'ten': '10'
                                }
        self.articles       = ['a',
                                 'an',
                                 'the'
                                ]


        self.periodStrip    = re.compile(r"(?!<=\d)(\.)(?!\d)") # Use raw string for regex
        self.commaStrip     = re.compile(r"(\d)(\,)(\d)")      # Use raw string for regex
        self.punct          = [';', r"/", '[', ']', '"', '{', '}',
                                 '(', ')', '=', '+', '\\', '_', '-',
                                 '>', '<', '@', '`', ',', '?', '!']


    def evaluate(self, quesIds=None):
        if quesIds == None:
            quesIds = [quesId for quesId in self.params['question_id']]
        gts = {}
        res = {}
        for quesId in quesIds:
            gts[quesId] = self.vqa.qa[quesId]
            res[quesId] = self.vqaRes.qa[quesId]

        # =================================================
        # Compute accuracy
        # =================================================
        accQA       = []
        accQuesType = {}
        accAnsType  = {}
        print("computing accuracy") # Fixed print
        step = 0
        for quesId in quesIds:
            # Fixed variable name `ansDic` to `ans_dict` for clarity
            for ans_dict in gts[quesId]['answers']:
                ans_dict['answer'] = ans_dict['answer'].replace('\n', ' ')
                ans_dict['answer'] = ans_dict['answer'].replace('\t', ' ')
                ans_dict['answer'] = ans_dict['answer'].strip()
            
            # Ensure res[quesId] is a dict and has 'answer' key before accessing
            if not isinstance(res[quesId], dict) or 'answer' not in res[quesId]:
                # Handle cases where a question might not have a prediction
                # You might log a warning or skip this question
                # print(f"Warning: No valid prediction found for quesId {quesId}. Skipping.")
                continue # Skip this question if no valid prediction

            resAns = res[quesId]['answer']
            resAns = resAns.replace('\n', ' ')
            resAns = resAns.replace('\t', ' ')
            resAns = resAns.strip()
            gtAcc = []
            gtAnswers = [ans['answer'] for ans in gts[quesId]['answers']]

            if len(set(gtAnswers)) > 1:
                for ans_dict in gts[quesId]['answers']:
                    ans_dict['answer'] = self.processPunctuation(ans_dict['answer'])
                    ans_dict['answer'] = self.processDigitArticle(ans_dict['answer'])
                resAns = self.processPunctuation(resAns)
                resAns = self.processDigitArticle(resAns)

            for gtAnsDatum in gts[quesId]['answers']:
                # The original `item!=gtAnsDatum` relies on object identity, which can be fragile.
                # Better to compare the 'answer' string itself.
                # Also, min(1, len(matchingAns)/3) is only if there are 10 annotators and 3 for full credit.
                # `self.n` is initialized to 2. Let's use `self.n` or the standard 3.0.
                
                # Standard VQAv2 accuracy calculation is `min(count_of_pred_in_gt / 3.0, 1.0)`
                # Where `count_of_pred_in_gt` is how many annotators gave the predicted answer.
                
                # Re-implementing the core accuracy logic:
                # Count how many annotators gave the exact `resAns` as an answer
                count_of_matching_answers = sum(1 for gt_ans_obj in gts[quesId]['answers'] if self.processDigitArticle(self.processPunctuation(gt_ans_obj['answer'])) == resAns)
                
                acc = float(count_of_matching_answers) / 3.0 # Divide by 3.0 as per official VQAv2 eval
                acc = min(acc, 1.0) # Cap at 1.0
                
                # The original loop `for gtAnsDatum in gts[quesId]['answers']` and then `min(1, float(len(matchingAns))/3)`
                # is not the standard VQAv2 soft accuracy. The soft accuracy applies to the *predicted* answer.
                # So we calculate one accuracy score per question.
                
                # This `gtAcc` list and `avgGTAcc` calculation is for the old VQA 1.0 way,
                # where each GT answer itself contributed to accuracy computation.
                # For VQA 2.0, it's about the predicted answer's match with multiple GTs.
                # Let's directly calculate `avgGTAcc` per question once.
                
                # The `acc` computed above is the accuracy for this specific question.
                # We should append `acc` directly to `accQA`.
                pass # Original loop will be replaced

            # Re-calculating avgGTAcc per question (based on the predicted answer's matches)
            # This logic should be outside the inner loop over gts[quesId]['answers']
            # if we are calculating one accuracy per question.
            
            # Find the accuracy for the `resAns` (predicted answer) against all ground truths
            count_of_pred_in_gt = sum(1 for gt_ans_obj in gts[quesId]['answers'] if self.processDigitArticle(self.processPunctuation(gt_ans_obj['answer'])) == resAns)
            avgGTAcc = float(count_of_pred_in_gt) / 3.0 # This is the core VQAv2 soft accuracy
            avgGTAcc = min(avgGTAcc, 1.0) # Cap at 1.0

            accQA.append(avgGTAcc) # This is the final accuracy for the current question

            quesType    = gts[quesId]['question_type']
            ansType     = gts[quesId]['answer_type']
            
            if quesType not in accQuesType:
                accQuesType[quesType] = []
            accQuesType[quesType].append(avgGTAcc)
            
            if ansType not in accAnsType:
                accAnsType[ansType] = []
            accAnsType[ansType].append(avgGTAcc)
            
            self.setEvalQA(quesId, avgGTAcc)
            self.setEvalQuesType(quesId, quesType, avgGTAcc)
            self.setEvalAnsType(quesId, ansType, avgGTAcc)
            
            if step % 100 == 0:
                self.updateProgress(step/float(len(quesIds)))
            step = step + 1

        self.setAccuracy(accQA, accQuesType, accAnsType)
        print("Done computing accuracy") # Fixed print

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) != None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = self.periodStrip.sub("",
                                         outText,
                                         re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            # else: # Removed else block if it was just pass, or keep if it's explicitly intended.
            #     pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = ' '.join(outText)
        return outText

    def setAccuracy(self, accQA, accQuesType, accAnsType):
        # Ensure accQA is not empty to avoid ZeroDivisionError
        overall_accuracy = round(100 * float(sum(accQA)) / len(accQA), self.n) if len(accQA) > 0 else 0.0
        self.accuracy['overall']        = overall_accuracy
        
        self.accuracy['perQuestionType'] = {quesType: round(100*float(sum(accQuesType[quesType]))/len(accQuesType[quesType]), self.n) 
                                           for quesType in accQuesType if len(accQuesType[quesType]) > 0}
        self.accuracy['perAnswerType']  = {ansType:  round(100*float(sum(accAnsType[ansType]))/len(accAnsType[ansType]), self.n) 
                                           for ansType in accAnsType if len(accAnsType[ansType]) > 0}


    def setEvalQA(self, quesId, acc):
        self.evalQA[quesId] = round(100*acc, self.n)

    def setEvalQuesType(self, quesId, quesType, acc):
        if quesType not in self.evalQuesType:
            self.evalQuesType[quesType] = {}
        self.evalQuesType[quesType][quesId] = round(100*acc, self.n)

    def setEvalAnsType(self, quesId, ansType, acc):
        if ansType not in self.evalAnsType:
            self.evalAnsType[ansType] = {}
        self.evalAnsType[ansType][quesId] = round(100*acc, self.n)

    def updateProgress(self, progress):
        barLength = 20
        status = ""
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
            status = "error: progress var must be float\r\n"
        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1
            status = "Done...\r\n"
        block = int(round(barLength*progress))
        text = "\rFinshed Percent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), int(progress*100), status)
        sys.stdout.write(text)
        sys.stdout.flush()
