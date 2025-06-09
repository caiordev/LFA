#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from automaton import FiniteAutomaton
from grammar import classify_word, CATEGORIES, is_linking_verb

class SyntacticAnalyzer:
    """
    Analisador sintático que utiliza autômatos finitos para identificar
    funções sintáticas básicas em frases do português brasileiro.
    """
    def __init__(self):
        self.automaton = self._build_automaton()
        
    def _build_automaton(self):
        """
        Constrói um autômato finito para análise sintática.
        Este autômato identifica funções sintáticas básicas como sujeito, verbo e objeto.
        """
        # Criar autômato para análise sintática
        automaton = FiniteAutomaton("SyntacticAnalyzer")
        
        # Estados do autômato
        # q0: Estado inicial
        # q1: Após reconhecer determinante do sujeito
        # q2: Após reconhecer sujeito
        # q3: Após reconhecer verbo
        # q4: Após reconhecer determinante do objeto
        # q5: Após reconhecer objeto ou advérbio
        # q6: Após reconhecer preposição
        # q7: Após reconhecer determinante do objeto indireto
        # q8: Após reconhecer objeto indireto
        # q9: Estado final (após pontuação)
        
        # Adicionar estados
        for i in range(10):
            is_final = (i == 9)  # Apenas q9 é estado final
            automaton.add_state(f"q{i}", is_final=is_final)
        
        # Definir estado inicial
        automaton.set_initial_state("q0")
        
        # Adicionar transições
        # Reconhecimento do sujeito
        automaton.add_transition("q0", "DET_SUJEITO", "q1")  # Determinante do sujeito
        automaton.add_transition("q0", "SUJEITO", "q2")      # Sujeito sem determinante
        automaton.add_transition("q1", "SUJEITO", "q2")      # Sujeito após determinante
        
        # Reconhecimento do verbo
        automaton.add_transition("q2", "VERBO", "q3")        # Verbo após sujeito
        
        # Reconhecimento do objeto direto, predicativo ou verbo no infinitivo
        automaton.add_transition("q3", "DET_OBJETO", "q4")   # Determinante do objeto
        automaton.add_transition("q3", "OBJETO", "q5")       # Objeto sem determinante (inclui adjetivos)
        automaton.add_transition("q3", "PREDICATIVO", "q5")  # Predicativo do sujeito após verbo de ligação
        automaton.add_transition("q3", "VERBO_INFINITIVO", "q5")  # Verbo no infinitivo após verbo auxiliar
        automaton.add_transition("q3", "ADVERBIO", "q5")     # Advérbio após verbo
        automaton.add_transition("q3", "PREPOSICAO", "q6")   # Preposição após verbo (para frases como "Eu liguei ao diretor")
        automaton.add_transition("q4", "OBJETO", "q5")       # Objeto após determinante
        
        # Reconhecimento do objeto indireto
        automaton.add_transition("q5", "PREPOSICAO", "q6")   # Preposição após objeto
        automaton.add_transition("q5", "ADVERBIO", "q5")     # Advérbio após objeto ou outro advérbio
        automaton.add_transition("q6", "DET_OBJ_IND", "q7")  # Determinante do objeto indireto
        automaton.add_transition("q6", "OBJ_INDIRETO", "q8") # Objeto indireto sem determinante
        automaton.add_transition("q7", "OBJ_INDIRETO", "q8") # Objeto indireto após determinante
        automaton.add_transition("q8", "ADVERBIO", "q8")     # Advérbio após objeto indireto
        
        # Finalização com pontuação
        automaton.add_transition("q3", "PONTUACAO", "q9")    # Frase termina após verbo
        automaton.add_transition("q5", "PONTUACAO", "q9")    # Frase termina após objeto direto, advérbio ou verbo no infinitivo
        automaton.add_transition("q8", "PONTUACAO", "q9")    # Frase termina após objeto indireto ou advérbio
        
        return automaton
    
    def _map_syntactic_functions(self, tokens):
        """
        Mapeia tokens para funções sintáticas com base em sua posição e classe gramatical.
        Esta é uma simplificação, pois a análise sintática real é muito mais complexa.
        """
        mapped_tokens = []
        state = 0  # Estado atual no processo de mapeamento
        
        for token, category in tokens:
            if state == 0:  # Esperando sujeito ou determinante do sujeito
                if category == "ARTIGO":
                    mapped_tokens.append((token, category, "DET_SUJEITO"))
                    state = 1
                elif category == "SUBSTANTIVO":
                    mapped_tokens.append((token, category, "SUJEITO"))
                    state = 2
                elif category == "PRONOME":
                    mapped_tokens.append((token, category, "SUJEITO"))
                    state = 2
                else:
                    mapped_tokens.append((token, category, "DESCONHECIDO"))
            
            elif state == 1:  # Após determinante do sujeito, esperando sujeito
                if category == "SUBSTANTIVO":
                    mapped_tokens.append((token, category, "SUJEITO"))
                    state = 2
                elif category == "ADJETIVO":
                    mapped_tokens.append((token, category, "MODIFICADOR_SUJEITO"))
                    # Continuamos no estado 1, esperando o substantivo
                else:
                    mapped_tokens.append((token, category, "DESCONHECIDO"))
            
            elif state == 2:  # Após sujeito, esperando verbo
                if category == "VERBO":
                    mapped_tokens.append((token, category, "VERBO"))
                    state = 3
                else:
                    mapped_tokens.append((token, category, "DESCONHECIDO"))
            
            elif state == 3:  # Após reconhecer o verbo, esperando objeto, preposição, advérbio ou pontuação
                if category == "ARTIGO":
                    mapped_tokens.append((token, category, "DET_OBJETO"))
                    state = 4
                elif category == "SUBSTANTIVO":
                    mapped_tokens.append((token, category, "OBJETO"))
                    state = 5
                elif category == "ADJETIVO" and any(t[2] == "VERBO" and is_linking_verb(t[0]) for t in mapped_tokens):
                    # Se o verbo anterior for de ligação, o adjetivo é predicativo do sujeito
                    mapped_tokens.append((token, category, "PREDICATIVO"))
                    state = 5
                elif category == "VERBO" and token.endswith('r'):  # Verbo no infinitivo
                    mapped_tokens.append((token, category, "VERBO_INFINITIVO"))
                    state = 5
                elif category == "PREPOSICAO":
                    mapped_tokens.append((token, category, "PREPOSICAO"))
                    state = 6
                elif category == "ADVERBIO":
                    mapped_tokens.append((token, category, "ADVERBIO"))
                    state = 5  # Consideramos que após um advérbio podemos ter pontuação
                elif category == "PONTUACAO":
                    mapped_tokens.append((token, category, "PONTUACAO"))
                    state = 9  # Estado final
                else:
                    mapped_tokens.append((token, category, "DESCONHECIDO"))
            
            elif state == 4:  # Após determinante do objeto, esperando objeto
                if category == "SUBSTANTIVO":
                    mapped_tokens.append((token, category, "OBJETO"))
                    state = 5
                elif category == "ADJETIVO":
                    mapped_tokens.append((token, category, "MODIFICADOR_OBJETO"))
                    # Continuamos no estado 4, esperando o substantivo
                else:
                    mapped_tokens.append((token, category, "DESCONHECIDO"))
            
            elif state == 5:  # Após reconhecer o objeto ou advérbio, esperando preposição, advérbio ou pontuação
                if category == "PREPOSICAO":
                    mapped_tokens.append((token, category, "PREPOSICAO"))
                    state = 6
                elif category == "ADVERBIO":
                    mapped_tokens.append((token, category, "ADVERBIO"))
                    # Continuamos no estado 5 pois após um advérbio podemos ter pontuação
                elif category == "PONTUACAO":
                    mapped_tokens.append((token, category, "PONTUACAO"))
                    state = 9  # Estado final
                else:
                    mapped_tokens.append((token, category, "DESCONHECIDO"))
            
            elif state == 6:  # Após reconhecer preposição, esperando determinante ou objeto indireto
                if category == "ARTIGO":
                    mapped_tokens.append((token, category, "DET_OBJ_IND"))
                    state = 7
                elif category == "SUBSTANTIVO":
                    mapped_tokens.append((token, category, "OBJ_INDIRETO"))
                    state = 8
                else:
                    mapped_tokens.append((token, category, "DESCONHECIDO"))
            
            elif state == 7:  # Após determinante do objeto indireto, esperando objeto indireto
                if category == "SUBSTANTIVO":
                    mapped_tokens.append((token, category, "OBJ_INDIRETO"))
                    state = 8
                elif category == "ADJETIVO":
                    mapped_tokens.append((token, category, "MODIFICADOR_OBJ_IND"))
                    # Continuamos no estado 7, esperando o substantivo
                else:
                    mapped_tokens.append((token, category, "DESCONHECIDO"))
            
            elif state == 8:  # Após reconhecer objeto indireto, esperando advérbio ou pontuação
                if category == "PONTUACAO":
                    mapped_tokens.append((token, category, "PONTUACAO"))
                    state = 9  # Estado final
                elif category == "ADVERBIO":
                    mapped_tokens.append((token, category, "ADVERBIO"))
                    # Após um advérbio, esperamos pontuação
                    state = 8
                else:
                    mapped_tokens.append((token, category, "DESCONHECIDO"))
            
            else:  # Estado desconhecido ou final
                mapped_tokens.append((token, category, "DESCONHECIDO"))
        
        return mapped_tokens
    
    def _tokenize_and_classify(self, sentence):
        """
        Tokeniza e classifica as palavras da frase em categorias gramaticais.
        """
        # Tokenização simples por espaços e remoção de pontuação
        tokens = []
        current_token = ""
        
        for char in sentence:
            if char.isalnum() or char in "áàâãéèêíìîóòôõúùûçÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ":
                current_token += char
            else:
                if current_token:
                    tokens.append(current_token.lower())
                    current_token = ""
                if not char.isspace():  # Se não for espaço, é pontuação
                    tokens.append(char)
        
        if current_token:  # Adicionar o último token se existir
            tokens.append(current_token.lower())
        
        # Classificar cada token
        classified_tokens = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            # Tratamento especial para contrações como "ao" (a + o)
            if token == "ao" or token == "aos":
                category = "PREPOSICAO"
            else:
                category = classify_word(token)
                if category is None:
                    category = "UNKNOWN"  # Categoria desconhecida
            
            classified_tokens.append((token, category))
            i += 1
        
        return classified_tokens
    
    def analyze_sentence(self, sentence):
        """
        Realiza a análise sintática da frase, identificando funções sintáticas básicas.
        """
        # Tokenizar e classificar as palavras
        classified_tokens = self._tokenize_and_classify(sentence)
        
        # Mapear para funções sintáticas
        syntactic_tokens = self._map_syntactic_functions(classified_tokens)
        
        # Verificar se a frase segue a estrutura sintática esperada
        # usando o autômato para validação
        automaton = self.automaton
        automaton.reset()
        
        valid = True
        syntactic_functions = [function for _, _, function in syntactic_tokens]
        
        # Rastrear o caminho percorrido no autômato
        path = [("q0", None)]  # Lista de tuplas (estado, símbolo)
        current_state = "q0"
        
        for function in syntactic_functions:
            if function != "DESCONHECIDO":
                if not automaton.process_input(function):
                    valid = False
                    break
                else:
                    # Atualizar o caminho
                    next_state = automaton.current_state.name
                    path.append((next_state, function))
                    current_state = next_state
        
        is_valid = valid and automaton.is_in_final_state()
        
        return is_valid, syntactic_tokens, path
    
    def explain_analysis(self, sentence):
        """
        Explica a análise sintática da frase, mostrando as funções sintáticas identificadas.
        """
        is_valid, syntactic_tokens, path = self.analyze_sentence(sentence)
        
        explanation = []
        explanation.append(f"Frase: {sentence}")
        
        explanation.append("\nAnálise Sintática:")
        for token, category, function in syntactic_tokens:
            explanation.append(f"  - '{token}': {category} → {function}")
        
        explanation.append("\nResultado da Análise:")
        if is_valid:
            explanation.append("  ✓ Estrutura sintática válida!")
            
            # Identificar as principais partes da oração
            sujeito = [token for token, _, function in syntactic_tokens 
                      if function in ["SUJEITO", "DET_SUJEITO", "MODIFICADOR_SUJEITO"]]
            predicado = [token for token, _, function in syntactic_tokens 
                        if function in ["VERBO", "VERBO_INFINITIVO", "OBJETO", "DET_OBJETO", "MODIFICADOR_OBJETO", 
                                       "PREPOSICAO", "OBJ_INDIRETO", "DET_OBJ_IND", "MODIFICADOR_OBJ_IND", "PREDICATIVO"]]
            
            explanation.append("\nPartes da Oração:")
            explanation.append(f"  - Sujeito: {' '.join(sujeito)}")
            explanation.append(f"  - Predicado: {' '.join(predicado)}")
            
            # Identificar componentes específicos
            verbo_auxiliar = [token for token, _, function in syntactic_tokens if function == "VERBO"]
            verbo_infinitivo = [token for token, _, function in syntactic_tokens if function == "VERBO_INFINITIVO"]
            
            # Combinar verbos para exibição
            verbos_combinados = verbo_auxiliar + verbo_infinitivo
            
            # Verificar se o verbo é de ligação
            is_linking_verb_present = False
            for token, _, function in syntactic_tokens:
                if function == "VERBO" and is_linking_verb(token):
                    is_linking_verb_present = True
                    break
            
            # Identificar predicativo do sujeito (apenas com verbos de ligação)
            predicativo = [token for token, _, function in syntactic_tokens if function == "PREDICATIVO"]
            
            # Identificar objeto direto (apenas com verbos que não são de ligação)
            objeto_direto = [token for token, _, function in syntactic_tokens 
                           if function in ["OBJETO", "DET_OBJETO", "MODIFICADOR_OBJETO"]] if not is_linking_verb_present else []
            
            objeto_indireto = [token for token, _, function in syntactic_tokens 
                             if function in ["OBJ_INDIRETO", "DET_OBJ_IND", "MODIFICADOR_OBJ_IND", "PREPOSICAO"]]
            
            if verbos_combinados:
                explanation.append(f"  - Verbo: {' '.join(verbos_combinados)}")
            if predicativo:
                explanation.append(f"  - Predicativo do Sujeito: {' '.join(predicativo)}")
            if objeto_direto:
                explanation.append(f"  - Objeto Direto: {' '.join(objeto_direto)}")
            if objeto_indireto:
                explanation.append(f"  - Objeto Indireto: {' '.join(objeto_indireto)}")
        else:
            explanation.append("  ✗ Estrutura sintática inválida! A frase não segue um padrão sintático reconhecido.")
        
        # Obter a visualização do autômato com o caminho percorrido
        automaton_visualization = self.visualize_automaton_path(sentence, syntactic_tokens, path)
        
        # Juntar a explicação com a visualização do autômato
        complete_explanation = "\n".join(explanation)
        automaton_viz_str = "\n".join(automaton_visualization)
        
        return complete_explanation, automaton_viz_str
    
    def get_automaton_description(self):
        """
        Retorna uma descrição do autômato utilizado para análise sintática.
        """
        description = []
        description.append("Autômato para Análise Sintática:")
        description.append("\nEstados:")
        description.append("  - q0: Estado inicial (início da frase)")
        description.append("  - q1: Após reconhecer determinante do sujeito")
        description.append("  - q2: Após reconhecer sujeito")
        description.append("  - q3: Após reconhecer verbo")
        description.append("  - q4: Após reconhecer determinante do objeto")
        description.append("  - q5: Após reconhecer objeto")
        description.append("  - q6: Após reconhecer preposição")
        description.append("  - q7: Após reconhecer determinante do objeto indireto")
        description.append("  - q8: Após reconhecer objeto indireto")
        description.append("  - q9: Estado final (após pontuação)")
        
        description.append("\nTransições:")
        description.append("  - q0 --DET_SUJEITO--> q1")
        description.append("  - q0 --SUJEITO--> q2")
        description.append("  - q1 --SUJEITO--> q2")
        description.append("  - q2 --VERBO--> q3")
        description.append("  - q3 --DET_OBJETO--> q4")
        description.append("  - q3 --OBJETO--> q5")
        description.append("  - q3 --PONTUACAO--> q9")
        description.append("  - q4 --OBJETO--> q5")
        description.append("  - q5 --PREPOSICAO--> q6")
        description.append("  - q5 --PONTUACAO--> q9")
        description.append("  - q6 --DET_OBJ_IND--> q7")
        description.append("  - q6 --OBJ_INDIRETO--> q8")
        description.append("  - q7 --OBJ_INDIRETO--> q8")
        description.append("  - q8 --PONTUACAO--> q9")
        
        description.append("\nRepresentação Visual:")
        description.append("""
        (q0) --DET_SUJEITO--> (q1) --SUJEITO--> (q2) --VERBO--> (q3) --DET_OBJETO--> (q4) --OBJETO--> (q5) --PREPOSICAO--> (q6) --DET_OBJ_IND--> (q7) --OBJ_INDIRETO--> (q8) --PONTUACAO--> (q9)*
          |                                        |                    |                                                           |
          |                                        |                    |                                                           |
          +--SUJEITO--> (q2)                       +--OBJETO--> (q5)    +--OBJ_INDIRETO--> (q8)                                    |
                                                     |                                                                             |
                                                     |                                                                             |
                                                     +--PONTUACAO--> (q9)*                                                         |
                                                                                                                                  |
                                                                                                                                  |
                                                                                                                                  v
                                                                                                                               (q9)*
                                                                                                                                               (q9)*
        """)
        
        description.append("\nExplicação:")
        description.append("  - Este autômato reconhece frases com a estrutura sintática básica do português brasileiro.")
        description.append("  - Ele identifica sujeito, verbo, objeto direto e objeto indireto.")
        description.append("  - O estado final q9 só é alcançado se a frase terminar com pontuação após uma estrutura válida.")
        description.append("  - As transições representam o reconhecimento de funções sintáticas específicas.")
        
        return "\n".join(description)
        
    def visualize_automaton_path(self, sentence, syntactic_tokens, path):
        """
        Visualiza o caminho percorrido no autômato durante a análise da frase.
        
        Args:
            sentence: A frase analisada
            syntactic_tokens: Lista de tuplas (token, categoria, função sintática)
            path: Lista de tuplas (estado, símbolo) representando o caminho percorrido
            
        Returns:
            Uma lista de strings representando a visualização do autômato
        """
        # Criar uma lista para armazenar a visualização
        visualization = []
        
        # Adicionar cabeçalho
        visualization.append("======================================================================")
        visualization.append("  AUTÔMATO PARA A FRASE ANALISADA")
        visualization.append("======================================================================")
        
        # Adicionar o caminho percorrido
        visualization.append("\nCaminho Percorrido no Autômato:")
        for i in range(1, len(path)):
            prev_state, _ = path[i-1]
            curr_state, symbol = path[i]
            visualization.append(f"  {prev_state} --{symbol}--> {curr_state}")
        
        # Identificar as principais partes da oração
        sujeito = [token for token, _, function in syntactic_tokens 
                 if function in ["SUJEITO", "DET_SUJEITO", "MODIFICADOR_SUJEITO"]]
        
        predicado = [token for token, _, function in syntactic_tokens 
                   if function not in ["SUJEITO", "DET_SUJEITO", "MODIFICADOR_SUJEITO", "PONTUACAO"]]
        
        verbos = [token for token, _, function in syntactic_tokens if function == "VERBO"]
        verbos_infinitivo = [token for token, _, function in syntactic_tokens if function == "VERBO_INFINITIVO"]
        verbos_combinados = verbos + verbos_infinitivo
        
        # Verificar se o verbo é de ligação
        is_linking_verb_present = False
        for token, _, function in syntactic_tokens:
            if function == "VERBO" and is_linking_verb(token):
                is_linking_verb_present = True
                break
        
        # Identificar predicativo do sujeito (apenas com verbos de ligação)
        predicativo = [token for token, _, function in syntactic_tokens if function == "PREDICATIVO"]
        
        # Identificar objeto direto (apenas com verbos que não são de ligação)
        objeto_direto = [token for token, _, function in syntactic_tokens 
                       if function in ["OBJETO", "DET_OBJETO", "MODIFICADOR_OBJETO"]] if not is_linking_verb_present else []
        
        objeto_indireto = [token for token, _, function in syntactic_tokens 
                         if function in ["OBJ_INDIRETO", "DET_OBJ_IND", "MODIFICADOR_OBJ_IND", "PREPOSICAO"]]
        
        # Identificar advérbios
        adverbios = [token for token, _, function in syntactic_tokens if function == "ADVERBIO"]
        
        # Adicionar as partes da oração à visualização
        visualization.append("\nPartes da Oração:")
        visualization.append(f"  - Sujeito: {' '.join(sujeito)}")
        visualization.append(f"  - Predicado: {' '.join(predicado)}")
        
        if verbos_combinados:
            visualization.append(f"  - Verbo: {' '.join(verbos_combinados)}")
        
        if predicativo:
            visualization.append(f"  - Predicativo do Sujeito: {' '.join(predicativo)}")
        
        if objeto_direto:
            visualization.append(f"  - Objeto Direto: {' '.join(objeto_direto)}")
        
        if objeto_indireto:
            visualization.append(f"  - Objeto Indireto: {' '.join(objeto_indireto)}")
        
        if adverbios:
            visualization.append(f"  - Advérbio: {' '.join(adverbios)}")
        
        # Adicionar a representação visual do autômato
        visualization.append("\nRepresentação Visual do Autômato (caminho destacado):")
        
        # Definir a representação base do autômato
        automaton_base = """
        (q0) --DET_SUJEITO--> (q1) --SUJEITO--> (q2) --VERBO--> (q3) --DET_OBJETO--> (q4) --OBJETO--> (q5) --PREPOSICAO--> (q6) --DET_OBJ_IND--> (q7) --OBJ_INDIRETO--> (q8) --PONTUACAO--> (q9)*
          |                                          |                    |                    |                                     |
          |                                          |                    |                    |                                     |
          +--SUJEITO--> (q2)                        +--OBJETO--> (q5)    +--OBJ_INDIRETO--> (q8)                                    |
                                                    |                    |                                                          |
                                                    |                    |                                                          |
                                                    +--ADVERBIO--> (q5)  +--ADVERBIO--> (q8)                                         |
                                                    |                                                                             |
                                                    |                                                                             |
                                                    +--PREPOSICAO--> (q6)                                                          |
                                                    |                                                                             |
                                                    |                                                                             |
                                                    +--PONTUACAO--> (q9)*                                                          |
                                                                                                                                  |
                                                                                                                                  |
                                                                                                                                  v
                                                                                                                               (q9)*
        """
        
        # Criar uma versão modificada do autômato base com o caminho destacado
        highlighted_automaton = automaton_base
        
        # Destacar os estados visitados
        for state, _ in path:
            # Substituir a representação normal do estado por uma destacada
            highlighted_automaton = highlighted_automaton.replace(f"({state})", f"[{state}]")
        
        # Destacar as transições percorridas
        for i in range(1, len(path)):
            prev_state, _ = path[i-1]
            curr_state, symbol = path[i]
            # Substituir a transição normal por uma destacada
            transition_str = f"{prev_state}) --{symbol}--> ({curr_state}"
            highlighted_transition = f"{prev_state}] --{symbol}--> [{curr_state}"
            highlighted_automaton = highlighted_automaton.replace(transition_str, highlighted_transition)
        
        visualization.append(highlighted_automaton)
        
        return visualization
