import random
import clingo
import time

import models_handler

class AspInterface:
	'''
	Parameters:
		- content: list with the program
	'''

	def __init__(self, 
		program_minimal_set : 'list[str]', 
		evidence : str, 
		asp_program : 'list[str]', 
		probabilistic_facts : 'dict[str,float]', 
		n_abducibles : int, 
		verbose : bool = False, 
		pedantic : bool = False, 
		n_samples : int = 1000
		) -> None:
		self.cautious_consequences : list[str] = []
		self.program_minimal_set : list[str] = sorted(set(program_minimal_set))
		self.asp_program : list[str] = sorted(set(asp_program))
		self.lower_probability_query : float = 0
		self.upper_probability_query : float = 0
		self.upper_probability_evidence : float = 0
		self.lower_probability_evidence : float = 0
		self.evidence : str = evidence
		# self.probabilistic_facts = probabilistic_facts # unused
		self.n_prob_facts : int = len(probabilistic_facts) # TODO: is probabilistic_facts used?
		self.n_abducibles : int = n_abducibles
		self.constraint_times_list : list[float] = []
		self.computed_models : int = 0
		self.grounding_time : float = 0
		self.n_worlds : int = 0
		self.world_analysis_time : float = 0
		self.computation_time : float = 0
		self.abductive_explanations : list[str] = []
		self.abduction_time : float = 0
		self.verbose : bool = verbose
		self.pedantic : bool = pedantic
		self.n_samples : int = n_samples
		self.prob_facts_dict : dict[str,float] = probabilistic_facts
		self.model_handler : models_handler.ModelsHandler = \
			models_handler.ModelsHandler(
				self.prob_facts_dict,
				self.evidence)

	
	def get_minimal_set_facts(self) -> float:
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
		ctl = clingo.Control(["--enum-mode=cautious", "-Wnone"])
		for clause in self.program_minimal_set:
			ctl.add('base',[],clause)

		ctl.ground([("base", [])])
		start_time = time.time()

		temp_cautious = []
		with ctl.solve(yield_=True) as handle:  # type: ignore
			for m in handle:  # type: ignore
				# i need only the last one
				temp_cautious = str(m).split(' ')  # type: ignore
			handle.get()  # type: ignore

		for el in temp_cautious:
			# if el != '' and (el.split(',')[-2] + ')' if el.count(',') > 0 else el.split('(')[0]) in self.probabilistic_facts:
			if el != '':
				self.cautious_consequences.append(el)

		# sys.exit()
		clingo_time = time.time() - start_time

		return clingo_time


	def compute_probabilities(self) -> None:
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
		ctl = clingo.Control(["0","--project","-Wnone"])
		for clause in self.asp_program:
			ctl.add('base',[],clause)

		if len(self.cautious_consequences) != 0:
			for c in self.cautious_consequences:
				ctl.add('base',[],":- not " + c + '.')
		
		start_time = time.time()
		ctl.ground([("base", [])])
		self.grounding_time = time.time() - start_time

		start_time = time.time()

		with ctl.solve(yield_=True) as handle:  # type: ignore
			for m in handle:  # type: ignore
				self.model_handler.add_value(str(m))  # type: ignore
				self.computed_models = self.computed_models + 1
			handle.get()   # type: ignore
		self.computation_time = time.time() - start_time

		# print(model_handler) # prints the models in world format

		start_time = time.time()
		self.lower_probability_query, self.upper_probability_query = self.model_handler.compute_lower_upper_probability()

		self.n_worlds = self.model_handler.get_number_worlds()

		self.world_analysis_time = time.time() - start_time


	def sample_world(self):
		'''
		Samples a world for approximate probability computation		
		'''
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


	def pick_random_index(self, block : int, id : str) -> list:
		'''
		Pick a random index, used in Gibbs sampling.
		TODO: this can be a static method.
		'''
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


	def mh_sampling(self) -> 'tuple[float, float]':
		'''
		MH sampling
		'''
		sampled = {}

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
			id = self.sample_world()

			if id in sampled:
				current_sampled = sampled[id][2]

				if random.random() < min(1, current_sampled/previous_sampled):
					k = k + 1
					n_upper = n_upper + sampled[id][0]
					n_lower = n_lower + sampled[id][1]

				previous_sampled = current_sampled
			else:
				i = 0
				for atm in ctl.symbolic_atoms:
					if atm.is_external:
						ctl.assign_external(atm.literal, True if id[i] == 'T' else False)
						i = i + 1

				upper = False
				lower = True
				sampled_evidence = False
				with ctl.solve(yield_=True) as handle:  # type: ignore
					for m in handle:  # type: ignore
						m1 = str(m).split(' ')  # type: ignore
						if 'e' in m1:
							sampled_evidence = True
							t_count = id.count('T')
							current_sampled = t_count if t_count > 0 else 1

							if random.random() < min(1, current_sampled/previous_sampled):
								k = k + 1
								if 'q' in m1:
									upper = True
								if "nq" in m1:
									lower = False

				if sampled_evidence is True:
					previous_sampled = current_sampled
				
				if upper:
					n_upper = n_upper + 1
					if sampled_evidence is True:
						sampled[id] = [1,0,current_sampled]
					if lower:
						n_lower = n_lower + 1
						if sampled_evidence is True:
							sampled[id] = [1,1,current_sampled]

		return n_lower/n_samples, n_upper/n_samples


	def gibbs_sampling(self, block: int) -> 'tuple[float, float]':
		'''
		Gibbs sampling
		'''
		# list of samples for the evidence
		sampled_evidence = {}
		# list of samples for the query
		sampled_query = {}

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
				if id in sampled_evidence:
					ev = sampled_evidence[id]
				else:
					i = 0
					for atm in ctl.symbolic_atoms:
						if atm.is_external:
							ctl.assign_external(atm.literal, True if id[i] == 'T' else False)
							i = i + 1
					with ctl.solve(yield_=True) as handle:  # type: ignore
						for m in handle:  # type: ignore
							m1 = str(m).split(' ')  # type: ignore
							if 'e' in m1:
								ev = True
								break

					sampled_evidence[id] = ev

			k = k + 1

			# Step 1: switch samples but keep the evidence true
			ev = False

			while ev is False:
				to_resample = self.pick_random_index(block, id)
				idNew = id
				# blocked gibbs
				if idNew in sampled_evidence:
					ev = sampled_evidence[idNew]
				else:
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
							
					sampled_evidence[idNew] = [ev]

			# step 2: ask query
			if idNew in sampled_query:
				n_upper = n_upper + sampled_query[idNew][0]
				n_lower = n_lower + sampled_query[idNew][1]
			else:
				i = 0
				for atm in ctl.symbolic_atoms:
					if atm.is_external:
						ctl.assign_external(atm.literal, True if idNew[i] == 'T' else False)
						i = i + 1

				upper = False
				lower = True
				with ctl.solve(yield_=True) as handle:  # type: ignore
					for m in handle:  # type: ignore
						m1 = str(m).split(' ')  # type: ignore
						if 'e' in m1:
							if 'q' in m1:
								upper = True
							if "nq" in m1:
								lower = False

				if upper:
					n_upper = n_upper + 1
					sampled_query[idNew] = [1,0]

					if lower:
						n_lower = n_lower + 1
						sampled_query[idNew] = [1,1]


		return n_lower/n_samples, n_upper/n_samples


	def rejection_sampling(self) -> 'tuple[float, float]':
		'''
		Rejection Sampling
		'''
		sampled = {}
		
		ctl = clingo.Control(["0", "--project"])
		for clause in self.asp_program:
			ctl.add('base', [], clause)
		ctl.ground([("base", [])])

		n_samples = self.n_samples

		n_upper = 0
		n_lower = 0
		k = 0

		while k < n_samples:
			ev_sampled = False
			id = self.sample_world()

			if id in sampled:
				k = k + 1
				n_upper = n_upper + sampled[id][0]
				n_lower = n_lower + sampled[id][1]
			else:
				i = 0
				for atm in ctl.symbolic_atoms:
					if atm.is_external:
						ctl.assign_external(atm.literal, True if id[i] == 'T' else False)
						i = i + 1

				upper = False
				lower = True
				with ctl.solve(yield_=True) as handle:  # type: ignore
					for m in handle:  # type: ignore
						m1 = str(m).split(' ')  # type: ignore
						if 'e' in m1:
							ev_sampled = True
							if 'q' in m1:
								upper = True
							if "nq" in m1:
								lower = False

				if ev_sampled is True:
					k = k + 1
					sampled[id] = [0, 0]

				if upper is True:
					n_upper = n_upper + 1
					sampled[id] = [1, 0]
					if lower is True:
						n_lower = n_lower + 1
						sampled[id] = [1, 1]

		return n_lower/n_samples, n_upper/n_samples


	def sample_query(self, bound : bool = False) -> 'tuple[float, float]':
		'''
		Samples the query self.n_samples times
		If bound is True, stops when either the number of samples taken k
		is greater than self.n_samples or 
		2 * 1.96 * math.sqrt(p * (1-p) / k) < 0.02
		'''
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
		k = 0

		if bound is True:
			import math

		while k < n_samples:
			id = self.sample_world()

			if id in sampled:
				n_upper = n_upper + sampled[id][0]
				n_lower = n_lower + sampled[id][1]
			else:
				i = 0
				for atm in ctl.symbolic_atoms:
					# atm.symbol.name # functor
					# atm.symbol.arguments[0].number # index
					if atm.is_external:
						# possible since dicts are ordered in Python 3.7+
						ctl.assign_external(atm.literal, True if id[i] == 'T' else False)
						i = i + 1

				upper = False
				lower = True
				with ctl.solve(yield_=True) as handle:  # type: ignore
					for m in handle:  # type: ignore
						if "q" == str(m):  # type: ignore
							upper = True
						elif "nq" == str(m):  # type: ignore
							lower = False
				
						handle.get()  # type: ignore

				if upper:
					n_upper = n_upper + 1
					sampled[id] = [1, 0]
					if lower:
						n_lower = n_lower + 1
						sampled[id] = [1, 1]
			k = k + 1
			
			if bound is True:
				p = n_lower / k
				# condition = 2 * 1.96 * math.sqrt(p * (1-p) / k) >= 0.02
				condition = 2 * 1.96 * math.sqrt(p * (1-p) / k) < 0.02
				if condition and n_lower > 5 and k - n_lower > 5 and k % 101 == 0:
					a = 2 * 1.96 * math.sqrt(p * (1-p) / k)
					break
		
		return n_lower/k, n_upper/k


	def abduction_iter(self, n_abd: int, previously_computed : list) -> 'tuple[str, float]':
		'''
		Loop for exact abduction
		'''
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
		# TODO: instead of, for each iteration, rewrite the whole program,
		# use multi-shot with Number

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

		with ctl.solve(yield_=True) as handle:  # type: ignore
			for m in handle:  # type: ignore
				# print(m)
				computed_models.append(str(m))  # type: ignore
				# n_models = n_models + 1
			handle.get()  # type: ignore

		computation_time = time.time() - start_time

		if self.verbose:
			print("time: " + str(computation_time))

		return computed_models, computation_time


	def abduction(self) -> None:
		'''
		Abduction
		'''
		result = []
		start_time = time.time()
		abducibles_list = []
		model_handler = models_handler.ModelsHandler(self.n_prob_facts, "")

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


	def log_infos(self) -> None:
		'''
		Log some execution details
		'''
		print("Computed models: " + str(self.computed_models))
		print("Considered worlds: " + str(self.n_worlds))
		print("Grounding time (s): " + str(self.grounding_time))
		print("Probability computation time (s): " + str(self.computation_time))
		print("World analysis time (s): " + str(self.world_analysis_time))
	

	def print_asp_program(self) -> None:
		'''
		Utility that prints the ASP program
		'''
		for el in self.asp_program:
			print(el)
		if len(self.cautious_consequences) != 0:
			for c in self.cautious_consequences:
				print(":- not " + c + '.')
