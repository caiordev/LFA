"""Implementação simplificada de uma Máquina de Turing multi-fita.

O módulo contém:

* TuringMachine — classe que interpreta uma descrição textual de
  transições e executa a máquina sobre uma fita de entrada.
* simulate — helper utilizado por Main.py para rodar a máquina em
  múltiplos processos.

As docstrings seguem as convenções PEP 257 e o estilo Google, de forma a
facilitar geração de documentação automática por ferramentas como
Sphinx ou pdoc.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing, time, random

class TuringMachine:
    """Máquina de Turing determinística de fita única.

    A lógica de transição é fornecida como texto, em que cada linha contém
    cinco campos:

    `current_state current_symbol new_symbol direction new_state`

    Comentários iniciados por ; e linhas em branco são ignorados. O símbolo
    `*` é curinga para qualquer símbolo. O caráter `_` representa
    blank.

    Attributes:
        transitions: Dicionário mapeando pares `(estado, símbolo)` para o
            tuple `(novo_símbolo, direção, novo_estado)`.
    """
    def _init_(self, logic: str):
        """Construtor.

        Args:
            logic: Descrição da máquina em formato de texto (veja a descrição
                da classe para detalhes do formato).
        """
        self.transitions = {}
        self.parse_logic(logic)

    def parse_logic(self, logic: str) -> None:
        """Analisa o texto de transições e preenche `self.transitions`.

        Args:
            logic: Texto contendo regras separadas por quebras de linha.
        """
        for line in logic.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith(';'):
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            current_state = parts[0]
            current_symbol = parts[1]
            new_symbol = parts[2]
            direction = parts[3]
            new_state = parts[4]

            self.transitions[(current_state, current_symbol)] = (new_symbol, direction, new_state)

    def run(self, input_tape: list[str], show_steps: bool = False) -> str:
        """Executa a máquina até atingir um estado halt.

        Args:
            input_tape: Lista de símbolos de entrada. Pode conter `*` como
                marcador de posição inicial da cabeça.
            show_steps: Quando `True` imprime cada passo para depuração.

        Returns:
            String com o conteúdo final da fita.
        """
        tape = list(input_tape)
        head_position = 0
        state = '0'
        step_count = 0

        if '*' in tape:
            head_position = tape.index('*')
            tape.pop(head_position)

        if show_steps:
            print("Iniciando a execução da Máquina de Turing...")
            print("Fita inicial: " + "".join(tape))
            print("Estado inicial: 0")
            print("Posição da cabeça: " + str(head_position))
            print("-" * 30)

        while not state.startswith('halt'):
            step_count += 1
            if step_count > 1000:
                state = 'halt-error-potential-infinite-loop'
                if show_steps:
                    print("Limite de passos atingido (1000). Interrompendo devido a potencial loop infinito.")
                break

            current_symbol = tape[head_position] if 0 <= head_position < len(tape) else '_'

            transition = None
            transition_rule = None

            if (state, current_symbol) in self.transitions:
                transition = self.transitions[(state, current_symbol)]
                transition_rule = f"({state}, {current_symbol}) -> ({transition[0]}, {transition[1]}, {transition[2]})"
            elif (state, '*') in self.transitions:
                transition = self.transitions[(state, '*')]
                transition_rule = f"({state}, *) -> ({transition[0]}, {transition[1]}, {transition[2]})"

            if transition:
                new_symbol, direction, new_state = transition

                if new_symbol != '*':
                    if 0 <= head_position < len(tape):
                        tape[head_position] = new_symbol
                    elif head_position < 0:
                        tape.insert(0, new_symbol)
                        head_position = 0
                    else:
                        tape.append(new_symbol)

                if direction == 'l':
                    head_position -= 1
                elif direction == 'r':
                    head_position += 1
                elif direction == '*':
                    pass

                state = new_state if new_state != '*' else state

                if show_steps:
                    tape_display = list(tape)
                    tape_display_str_parts = []
                    for i, symbol in enumerate(tape_display):
                        if i == head_position:
                            tape_display_str_parts.append(f"[{symbol}]")
                        else:
                            tape_display_str_parts.append(symbol)
                    tape_display_str = "".join(tape_display_str_parts)

                    print(f"Passo: {step_count}")
                    print(f"Estado: {state}")
                    print(f"Fita: {tape_display_str}")
                    print(f"Posição da cabeça: {head_position}")
                    print(f"Símbolo atual: {current_symbol if current_symbol != '' else ''}")
                    print(f"Transição: {transition_rule}")
                    print("-" * 30)

            else:
                state = 'halt-error-no-transition'
                if show_steps:
                    print(f"Passo: {step_count}")
                    print(f"Estado: {state}")
                    print(f"Fita: {''.join(tape)}")
                    print(f"Posição da cabeça: {head_position}")
                    print(f"Símbolo atual: {current_symbol if current_symbol != '' else ''}")
                    print("Nenhuma transição encontrada para o estado e símbolo atuais. Interrompendo.")
                    print("-" * 30)
                break

            if head_position < 0:
                tape.insert(0, '_')
                head_position = 0
            elif head_position >= len(tape):
                tape.append('_')

        while tape and tape[0] == '_':
            tape.pop(0)
        while tape and tape[-1] == '_':
            tape.pop()

        if show_steps:
            if state.startswith('halt') and not state.startswith('halt-error'):
                print("Máquina de Turing interrompida com sucesso!")
            elif state.startswith('halt-error'):
                print(f"Máquina de Turing interrompida com estado de erro: {state}")
            print("Fita final: " + "".join(tape))
            print("-" * 30)

        return "".join(tape)

tm_logic = """
; Multiplication of two binary numbers, in tuple syntax.
; Example input: 11S10
; Symbol '_' is used for blank, 'S' is used instead of '*' as a literal separator.

; -- start state: start
0 0 0 l init       ; [0,1]: {L: init}
0 1 1 l init

; -- init
init _ + r right       ; ' ' => '_'; {write: '+', R: right}

; -- right
right 0 0 r right      ; [0,1,'']: R  => substituí '' por 'S'
right 1 1 r right
right S S r right
right _ _ l readB      ; ' ' => '_'; {L: readB}

; -- readB
readB 0 _ l doubleL    ; 0: {write: ' ', L: doubleL} => '_' no lugar de espaço
readB 1 _ l addA       ; 1: {write: ' ', L: addA}

; -- addA
addA 0 0 l addA        ; [0,1]: L => permanece em addA
addA 1 1 l addA
addA S S l read        ; '': {L: read} => substituído '' por 'S'

; -- doubleL
doubleL 0 0 l doubleL  ; [0,1]: L => continua em doubleL
doubleL 1 1 l doubleL
doubleL S 0 r shift    ; '': {write: 0, R: shift} => substituí '' por 'S'

; -- double
double 0 0 r double    ; [0,1,+]: R => permanece em double
double 1 1 r double
double + + r double
double S 0 r shift     ; '': {write: 0, R: shift} => substituí '' por 'S'

; -- shift
shift 0 S r shift0     ; 0: {write: '', R: shift0} => substituí '' por 'S'
shift 1 S r shift1     ; 1: {write: '*', R: shift1}
shift _ _ l tidy       ; ' ': {L: tidy} => '_' para branco

; -- shift0
shift0 0 0 r shift0    ; 0: {R: shift0}
shift0 1 0 r shift1    ; 1: {write: 0, R: shift1}
shift0 _ 0 r right     ; ' ' => '_'; {write: 0, R: right}

; -- shift1
shift1 0 1 r shift0    ; 0: {write: 1, R: shift0}
shift1 1 1 r shift1    ; 1: {R: shift1}
shift1 _ 1 r right     ; ' ' => '_'; {write: 1, R: right}

; -- tidy
tidy 0 _ l tidy        ; [0,1]: {write: ' ', L} => '_' para branco, permanece em tidy
tidy 1 _ l tidy
tidy + _ l halt-done   ; +: {write: ' ', L: done} => renomeei done para halt-done

; -- (done) => trocado para 'halt-done' para encerrar a máquina

; -- read
read 0 c l have0       ; 0: {write: c, L: have0}
read 1 c l have1       ; 1: {write: c, L: have1}
read + + l rewrite     ; +: {L: rewrite} (mantém '+')

; -- have0
have0 0 0 l have0      ; [0,1]: L => fica em have0
have0 1 1 l have0
have0 + + l add0       ; +: {L: add0}

; -- have1
have1 0 0 l have1
have1 1 1 l have1
have1 + + l add1

; -- add0
add0 0 O r back0       ; [0,' ']: {write: O, R: back0} => substitui ' ' por '_'
add0 _ O r back0
add0 1 I r back0       ; 1 => {write: I, R: back0}
add0 O O l add0        ; [O,I]: L => permanece
add0 I I l add0

; -- add1
add1 0 I r back1       ; [0,' ']: {write: I, R: back1}
add1 _ I r back1
add1 1 O l carry       ; 1 => {write: O, L: carry}
add1 O O l add1        ; [O,I] => L => continua
add1 I I l add1

; -- carry
carry 0 1 r back1      ; [0,' ']: {write: 1, R: back1}
carry _ 1 r back1
carry 1 0 l carry      ; 1 => {write: 0, L} => permanece em carry

; -- back0
back0 0 0 r back0      ; [0,1,O,I,+]: R => continua
back0 1 1 r back0
back0 O O r back0
back0 I I r back0
back0 + + r back0
back0 c 0 l read       ; c => {write: 0, L: read}

; -- back1
back1 0 0 r back1
back1 1 1 r back1
back1 O O r back1
back1 I I r back1
back1 + + r back1
back1 c 1 l read       ; c => {write: 1, L: read}

; -- rewrite
rewrite O 0 l rewrite  ; O => {write: 0, L}
rewrite I 1 l rewrite  ; I => {write: 1, L}
rewrite 0 0 l rewrite  ; 0 => {write: 0, L}
rewrite 1 1 l rewrite  ; 1 => {write: 1, L}
rewrite _ _ r double   ; ' ' => '_' => {R: double}

"""

def simulate(pair: tuple[str, str]) -> tuple[str, str, str]:
    """Wrapper para execução paralela via `ProcessPoolExecutor`.

    Args:
        pair: Dupla de strings binárias correspondentes aos fatores.

    Returns:
        Tuple `(input1, input2, resultado)` onde `resultado` é a
        multiplicação binária de `input1` por `input2`.
    """

    input1, input2 = pair
    tm = TuringMachine(tm_logic)  # cada processo tem sua própria instância
    tape = list(f"{input1}S{input2}_")
    result_tape = tm.run(tape, show_steps=False)
    return input1, input2, result_tape