#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from automaton import FiniteAutomaton
from grammar import classify_word, VALID_STRUCTURES

class SequentialStructureAnalyzer:
    """
    Analisador de estrutura sequencial que utiliza autômatos finitos para validar
    a ordem das classes gramaticais em frases do português brasileiro.
    
    Nota: Este analisador NÃO realiza análise sintática completa (que identificaria
    funções sintáticas como sujeito, predicado, objeto direto, etc.), mas apenas
    verifica se a sequência de classes gramaticais segue padrões válidos.
    """
    def __init__(self):
        self.automata = {}
        self._build_automata()
    
    def _build_automata(self):
        """
        Constrói os autômatos para cada estrutura sequencial válida definida na gramática.
        Cada autômato representa um padrão de sequência de classes gramaticais válido.
        """
        for i, structure in enumerate(VALID_STRUCTURES):
            automaton = FiniteAutomaton(f"Structure_{i}")
            
            # Criar estados para cada posição na estrutura
            for j in range(len(structure) + 1):
                is_final = (j == len(structure))
                automaton.add_state(f"q{j}", is_final=is_final)
            
            # Definir estado inicial
            automaton.set_initial_state("q0")
            
            # Adicionar transições
            for j in range(len(structure)):
                category = structure[j]
                automaton.add_transition(f"q{j}", category, f"q{j+1}")
            
            self.automata[i] = automaton
    
    def _tokenize_and_classify(self, sentence):
        """
        Método interno para tokenizar e classificar as palavras da frase.
        Este método é usado apenas internamente para preparar os dados para análise da estrutura sequencial.
        """
        # Remover espaços extras e dividir a frase em palavras
        words = sentence.strip().split()
        
        # Separar pontuação das palavras
        tokens = []
        for word in words:
            # Verificar se a palavra termina com pontuação
            if word and word[-1] in ".!?,;:":
                tokens.append(word[:-1])
                tokens.append(word[-1])
            else:
                tokens.append(word)
        
        # Classificar cada token em sua classe gramatical
        classified_tokens = []
        for token in tokens:
            if not token:  # Ignorar tokens vazios
                continue
                
            category = classify_word(token)
            if category:
                classified_tokens.append((token, category))
            else:
                classified_tokens.append((token, "UNKNOWN"))
        
        return classified_tokens
    
    def analyze_structure(self, sentence):
        """
        Realiza a análise da estrutura sequencial da frase, verificando se a ordem
        das classes gramaticais segue um padrão válido no português brasileiro.
        """
        # Tokenizar e classificar as palavras (apenas para uso interno)
        classified_tokens = self._tokenize_and_classify(sentence)
        
        # Extrair apenas as categorias gramaticais para validação
        categories = [category for _, category in classified_tokens]
        
        # Verificar se a sequência de categorias é aceita por algum autômato
        for structure_id, automaton in self.automata.items():
            automaton.reset()
            
            # Processar cada categoria como um símbolo de entrada para o autômato
            valid = True
            for category in categories:
                if not automaton.process_input(category):
                    valid = False
                    break
            
            # Verificar se o autômato terminou em um estado final
            if valid and automaton.is_in_final_state():
                return True, VALID_STRUCTURES[structure_id], classified_tokens
        
        return False, None, classified_tokens
    
    def explain_analysis(self, sentence):
        """
        Explica o resultado da análise da estrutura sequencial da frase.
        """
        valid, structure, classified_tokens = self.analyze_structure(sentence)
        
        explanation = []
        explanation.append(f"Frase: {sentence}")
        
        # Mostrar a estrutura sequencial encontrada
        explanation.append("\nAnálise da Estrutura Sequencial:")
        if valid:
            explanation.append(f"  ✓ Estrutura sequencial válida! Corresponde ao padrão: {' + '.join(structure)}")
            
            # Mostrar como a frase se encaixa no padrão
            explanation.append("\nComo a frase se encaixa no padrão:")
            for (token, category), pattern_part in zip(classified_tokens, structure):
                explanation.append(f"  - '{token}' → {category} ({pattern_part})")
        else:
            explanation.append("  ✗ Estrutura sequencial inválida! Não corresponde a nenhum padrão de sequência válido.")
            
            # Sugerir possíveis estruturas válidas
            explanation.append("\nAlguns padrões sequenciais válidos no português brasileiro:")
            # Mostrar apenas alguns exemplos para não sobrecarregar a saída
            for i, struct in enumerate(VALID_STRUCTURES[:5]):
                explanation.append(f"  {i+1}. {' + '.join(struct)}")
            if len(VALID_STRUCTURES) > 5:
                explanation.append(f"  ... e mais {len(VALID_STRUCTURES) - 5} padrões.")
        
        return "\n".join(explanation)
