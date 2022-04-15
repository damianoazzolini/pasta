import random
import clingo
from typing import Union
import time

# local
import models_handler

class AspInterface:
	'''
	Parameters:
		- content: list with the program
	'''

	def __init__(self, program_minimal_set: list, evidence: list, asp_program : list, probabilistic_facts : dict, n_abducibles : int, precision : int = 3, verbose : bool = False, pedantic = False, n_samples = 1000, prob_facts_dict = None) -> None:
		self.cautious_consequences : list = []
		self.program_minimal_set : list = sorted(set(program_minimal_set))
		self.asp_program : list = sorted(set(asp_program))
		self.lower_probability_query : int = 0
		self.upper_probability_query : int = 0
		self.upper_probability_evidence : int = 0
		self.lower_probability_evidence : int = 0
		self.precision : int = precision
		self.evidence : str = evidence
		# self.probabilistic_facts = probabilistic_facts # unused
		self.n_prob_facts : int = len(probabilistic_facts)
		self.n_abducibles : int = n_abducibles
		self.constraint_times_list : list = []
		self.computed_models : int = 0
		self.grounding_time : int = 0
		self.n_worlds : int = 0
		self.world_analysis_time : int = 0
		self.computation_time : int = 0
		self.abductive_explanations : list = []
		self.abduction_time : int = 0
		self.verbose : bool = verbose
		self.pedantic : bool = pedantic
		self.n_samples : int = n_samples
		self.prob_facts_dict : dict = prob_facts_dict

	'''
	Parameters:
		- None
	Return:
		- str
	Behavior:
		compute the minimal set of facts
		needed to make the query true. This operation is performed
		only if there is not evidence.
		Cautious consequences
		clingo <filename> -e cautious
	'''
	def get_minimal_set_facts(self) -> float:
		ctl = clingo.Control(["--enum-mode=cautious"])
		for clause in self.program_minimal_set:
			ctl.add('base',[],clause)

		ctl.ground([("base", [])])
		start_time = time.time()

		temp_cautious = []
		with ctl.solve(yield_=True) as handle:
			for m in handle:
				# i need only the last one
				temp_cautious = str(m).split(' ')
			handle.get()

		for el in temp_cautious:
			# if el != '' and (el.split(',')[-2] + ')' if el.count(',') > 0 else el.split('(')[0]) in self.probabilistic_facts:
			if el != '':
				self.cautious_consequences.append(el)

		# sys.exit()
		clingo_time = time.time() - start_time

		return clingo_time

	'''
	Parameters:
		- None
	Return:
		- int: number of computed models
		- float: grounding time
		- float: computing probability time
	Behavior:
		compute the lower and upper bound for the query
		clingo 0 <filename> --project
	'''
	def compute_probabilities(self) -> None:
		ctl = clingo.Control(["0","--project"])
		for clause in self.asp_program:
			ctl.add('base',[],clause)

		if len(self.cautious_consequences) != 0:
			for c in self.cautious_consequences:
				ctl.add('base',[],":- not " + c + '.')
		
		start_time = time.time()
		ctl.ground([("base", [])])
		self.grounding_time = time.time() - start_time

		start_time = time.time()
		model_handler = models_handler.ModelsHandler(self.precision, self.n_prob_facts, self.evidence)

		with ctl.solve(yield_=True) as handle:
			for m in handle:
				model_handler.add_value(str(m))
				self.computed_models = self.computed_models + 1
			handle.get()
		self.computation_time = time.time() - start_time

		# print(model_handler) # prints the models in world format

		start_time = time.time()
		self.lower_probability_query, self.upper_probability_query = model_handler.compute_lower_upper_probability()

		self.n_worlds = model_handler.get_number_worlds()

		self.world_analysis_time = time.time() - start_time

	# samples a world for approximate probability computation 
	def sample_world(self):
		id = ""
		# samples = []
		for key in self.prob_facts_dict:
			if random.random() < self.prob_facts_dict[key]:
				id = id + "T"
				# samples.append(True)
			else:
				id = id + "F"
				# samples.append(False)

		return id

	# this can be a static method
	def pick_random_index(self, block : int, id : str) -> list:
			# i = random.randint(0,len(id) - 1)
			# while i == 1:
			# 	i = random.randint(0,len(id) - 1)
		return sorted(set([random.randint(0, len(id) - 1) for i in range(0, block)]))

	def resample(self, i : int) -> str:
		for k in self.prob_facts_dict:
			key = k
			i = i - 1
			if i < 0:
				break

		if random.random() < self.prob_facts_dict[key]:
			return 'T'
		else:
			return 'F'

	'''
	MH sampling
	'''
	def mh_sampling(self) -> Union[float, float]:
		ctl = clingo.Control(["0", "--project"])
		for clause in self.asp_program:
			ctl.add('base', [], clause)
		ctl.ground([("base", [])])

		n_samples = self.n_samples

		n_upper = 0
		n_lower = 0

		# step 0: build initial sample
		id = self.sample_world()
		t_count = id.count('T')
		previous_sampled = t_count if t_count > 0 else 1

		k = 0
		while k < n_samples:
			# for k in range(0, n_samples):
			id = self.sample_world()
			i = 0
			for atm in ctl.symbolic_atoms:
				if atm.is_external:
					ctl.assign_external(atm.literal, True if id[i] == 'T' else False)
					i = i + 1

			upper = False
			lower = True
			with ctl.solve(yield_=True) as handle:
				for m in handle:
					m1 = str(m).split(' ')
					if 'e' in m1:
						k = k + 1
						t_count = id.count('T')
						current_sampled = t_count if t_count > 0 else 1

						if 'q' in m1 and random.random() < min(1, current_sampled/previous_sampled):
							upper = True
							if "nq" in m1:
								lower = False

			if upper:
				n_upper = n_upper + 1
				if lower:
					n_lower = n_lower + 1

		return n_lower/n_samples, n_upper/n_samples


	'''
	Gibbs sampling
	'''

	def gibbs_sampling(self, block: int) -> Union[float, float]:
		ctl = clingo.Control(["0", "--project"])
		for clause in self.asp_program:
			ctl.add('base', [], clause)
		ctl.ground([("base", [])])

		n_samples = self.n_samples

		n_upper = 0
		n_lower = 0
		k = 0

		while k < n_samples:
			# Step 0: sample evidence
			ev = False
			while ev is False:
				id = self.sample_world()
				i = 0
				for atm in ctl.symbolic_atoms:
					if atm.is_external:
						ctl.assign_external(atm.literal, True if id[i] == 'T' else False)
						i = i + 1
				with ctl.solve(yield_=True) as handle:
					for m in handle:
						m1 = str(m).split(' ')
						if 'e' in m1:
							ev = True
							break

			# non devo mantenere la lista di campioni qui
			k = k + 1

			# Step 1: switch samples but keep the evidence true
			ev = False
			while ev is False:
				to_resample = self.pick_random_index(block, id)
				idNew = id
				# blocked gibbs
				for i in to_resample:
					idNew = idNew[:i] + self.resample(i) + idNew[i+1:]

				i = 0
				for atm in ctl.symbolic_atoms:
					if atm.is_external:
						ctl.assign_external(atm.literal, True if idNew[i] == 'T' else False)
						i = i + 1
				with ctl.solve(yield_=True) as handle:
					for m in handle:
						m1 = str(m).split(' ')
						if 'e' in m1:
							ev = True
							break

			i = 0
			for atm in ctl.symbolic_atoms:
				if atm.is_external:
					ctl.assign_external(atm.literal, True if idNew[i] == 'T' else False)
					i = i + 1

			upper = False
			lower = True
			with ctl.solve(yield_=True) as handle:
				for m in handle:
					m1 = str(m).split(' ')
					if 'e' in m1:
						# k = k + 1
						if 'q' in m1:
							upper = True
							if "nq" in m1:
								lower = False

			if upper:
				n_upper = n_upper + 1
				if lower:
					n_lower = n_lower + 1

		return n_lower/n_samples, n_upper/n_samples

	'''
	Rejection Sampling
	'''
	def rejection_sampling(self) -> Union[float, float]:
		sampled = {}
		
		ctl = clingo.Control(["0", "--project"])
		for clause in self.asp_program:
			ctl.add('base', [], clause)
		ctl.ground([("base", [])])

		# n_bool_vars = self.n_prob_facts
		n_samples = self.n_samples

		n_upper = 0
		n_lower = 0
		k = 0

		while k < n_samples:
			id = self.sample_world()

			if id in sampled:
				k = k + 1
				n_upper = n_upper + sampled[id][0]
				n_lower = n_lower + sampled[id][1]
			else:
				i = 0
				for atm in ctl.symbolic_atoms:
					# atm.symbol.name # qui ho il funtore
					# atm.symbol.name # qui ho il funtore
					# atm.symbol.arguments[0].number # qui ho l'indice
					if atm.is_external:
						# posso fare questo perché i dizionari sono ordinati in Python 3.7+
						ctl.assign_external(atm.literal, True if id[i] == 'T' else False)
						i = i + 1

				upper = False
				lower = True
				with ctl.solve(yield_=True) as handle:
					for m in handle:
						m1 = str(m).split(' ')
						if 'e' in m1:
							k = k + 1
							if 'q' in m1:
								upper = True
								if "nq" in m1:
									lower = False

				if upper:
					n_upper = n_upper + 1
					sampled[id] = [1, 0]
					if lower:
						n_lower = n_lower + 1
						sampled[id] = [1, 1]

		return n_lower/n_samples, n_upper/n_samples

	def sample_query(self) -> Union[float, float]:
		# sampled worlds
		sampled = {}
		
		ctl = clingo.Control(["0", "--project"])
		for clause in self.asp_program:
			ctl.add('base', [], clause)
		ctl.ground([("base", [])])

		# n_bool_vars = self.n_prob_facts
		n_samples = self.n_samples

		n_upper = 0
		n_lower = 0
		for _ in range(0, n_samples):
			id = self.sample_world()

			if id in sampled:
				n_upper = n_upper + sampled[id][0]
				n_lower = n_lower + sampled[id][1]
			else:
				i = 0
				for atm in ctl.symbolic_atoms:
					# atm.symbol.name # qui ho il funtore
					# atm.symbol.name # qui ho il funtore
					# atm.symbol.arguments[0].number # qui ho l'indice
					if atm.is_external:
						# posso fare questo perché i dizionari sono ordinati in Python 3.7+
						ctl.assign_external(atm.literal, True if id[i] == 'T' else False)
						i = i + 1

				# res = []
				upper = False
				lower = True
				with ctl.solve(yield_=True) as handle:
					for m in handle:
						# res.append(str(m))
						if "q" == str(m):
							upper = True
						elif "nq" == str(m):
							lower = False
				
						handle.get()

				if upper:
					n_upper = n_upper + 1
					sampled[id] = [1, 0]
					if lower:
						n_lower = n_lower + 1
						sampled[id] = [1, 1]

		return n_lower/n_samples, n_upper/n_samples

	# loop for exact abduction
	def abduction_iter(self, n_abd: int, previously_computed : list) -> Union[str, float]:
		if self.verbose:
			print(str(n_abd) + " abd")

		ctl = clingo.Control(["0", "--project"])
		for clause in self.asp_program:
			ctl.add('base', [], clause)

		if len(self.cautious_consequences) != 0:
			for c in self.cautious_consequences:
				ctl.add('base', [], ":- not " + c + '.')

		if self.n_prob_facts == 0:
			ctl.add('base', [], ':- not q.')
		ctl.add('base', [], 'abd_facts_counter(C):- #count{X : abd_fact(X)} = C.')
		ctl.add('base', [], ':- abd_facts_counter(C), C != ' + str(n_abd) + '.')

		for exp in previously_computed:
			s = ":- "
			for el in exp:
				if el != "q" and not el.startswith('not_abd'):
					s = s + el + ","
			s = s[:-1] + '.'
			ctl.add('base', [], s)

		start_time = time.time()
		ctl.ground([("base", [])])
		self.grounding_time = time.time() - start_time

		computed_models = []

		with ctl.solve(yield_=True) as handle:
			for m in handle:
				# print(m)
				computed_models.append(str(m))
				# n_models = n_models + 1
			handle.get()

		computation_time = time.time() - start_time

		if self.verbose:
			print("time: " + str(computation_time))

		return computed_models, computation_time

	'''
	Abduction
	'''
	def abduction(self):
		result = []
		start_time = time.time()
		abducibles_list = []
		model_handler = models_handler.ModelsHandler(self.precision, self.n_prob_facts, None)

		for i in range(0, self.n_abducibles + 1):
			currently_computed, exec_time = self.abduction_iter(i, abducibles_list)
			self.computed_models = self.computed_models + len(currently_computed)
			if self.verbose:
				print("Models with " + str(i) + " abducibles: " + str(len(currently_computed)))
				if self.pedantic:
					print(currently_computed)

			# TODO: gestire len(currently_computed) > 0 and i == 0 (vero senza abducibili)

			if self.n_prob_facts == 0:
				# currently computed: list of computed models
				for i in range(0,len(currently_computed)):
					currently_computed[i] = currently_computed[i].split(' ')
					result.append(currently_computed[i])
				
				self.computed_models = self.computed_models + len(currently_computed)

				for cc in currently_computed:
					abducibles_list.append(cc)

			else:
				for el in currently_computed:
					model_handler.add_model_abduction(str(el))

				self.lower_probability_query, self.upper_probability_query = model_handler.compute_lower_upper_probability()

			# keep the best model
			self.lower_probability_query, self.upper_probability_query = model_handler.keep_best_model()
			self.constraint_times_list.append(exec_time)

		for el in model_handler.abd_worlds_dict:
			result.append(el[1:].split(' '))

		self.abduction_time = time.time() - start_time
		self.abductive_explanations = result

	
	# prints the ASP program
	def print_asp_program(self) -> None:
		for el in self.asp_program:
			print(el)
		if len(self.cautious_consequences) != 0:
			for c in self.cautious_consequences:
				print(":- not " + c + '.')
