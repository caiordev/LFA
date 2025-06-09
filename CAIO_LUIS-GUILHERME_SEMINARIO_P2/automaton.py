#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class State:
    """
    Classe que representa um estado do autômato finito.
    """
    def __init__(self, name, is_final=False):
        self.name = name
        self.is_final = is_final
        self.transitions = {}  # {símbolo: estado_destino}
    
    def add_transition(self, symbol, target_state):
        """
        Adiciona uma transição do estado atual para o estado alvo usando o símbolo especificado.
        """
        self.transitions[symbol] = target_state
    
    def get_next_state(self, symbol):
        """
        Retorna o próximo estado com base no símbolo de entrada.
        """
        return self.transitions.get(symbol)
    
    def __str__(self):
        return f"State({self.name}, final={self.is_final})"


class FiniteAutomaton:
    """
    Implementação de um Autômato Finito Determinístico (AFD).
    """
    def __init__(self, name):
        self.name = name
        self.states = {}  # {nome_estado: objeto_estado}
        self.initial_state = None
        self.current_state = None
    
    def add_state(self, name, is_final=False):
        """
        Adiciona um estado ao autômato.
        """
        state = State(name, is_final)
        self.states[name] = state
        return state
    
    def set_initial_state(self, state_name):
        """
        Define o estado inicial do autômato.
        """
        if state_name in self.states:
            self.initial_state = self.states[state_name]
            self.current_state = self.initial_state
        else:
            raise ValueError(f"Estado {state_name} não existe no autômato")
    
    def add_transition(self, from_state, symbol, to_state):
        """
        Adiciona uma transição entre estados.
        """
        if from_state in self.states and to_state in self.states:
            self.states[from_state].add_transition(symbol, self.states[to_state])
        else:
            missing = from_state if from_state not in self.states else to_state
            raise ValueError(f"Estado {missing} não existe no autômato")
    
    def reset(self):
        """
        Reinicia o autômato para o estado inicial.
        """
        self.current_state = self.initial_state
    
    def process_input(self, input_symbol):
        """
        Processa um símbolo de entrada e atualiza o estado atual.
        Retorna True se a transição foi bem-sucedida, False caso contrário.
        """
        if self.current_state is None:
            return False
        
        next_state = self.current_state.get_next_state(input_symbol)
        if next_state is not None:
            self.current_state = next_state
            return True
        return False
    
    def is_in_final_state(self):
        """
        Verifica se o autômato está em um estado final.
        """
        return self.current_state is not None and self.current_state.is_final
    
    def process_string(self, input_string):
        """
        Processa uma string completa e verifica se ela é aceita pelo autômato.
        """
        self.reset()
        
        for symbol in input_string:
            if not self.process_input(symbol):
                return False
        
        return self.is_in_final_state()
    
    def __str__(self):
        states_str = ", ".join([f"{name}{'*' if state.is_final else ''}" 
                               for name, state in self.states.items()])
        initial = self.initial_state.name if self.initial_state else "None"
        return f"Automaton({self.name}, states=[{states_str}], initial={initial})"
