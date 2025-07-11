"""Main module for benchmarking da Máquina de Turing Paralela.

Este módulo fornece funções utilitárias para:

1. Gerar números binários aleatórios.
2. Medir desempenho de multiplicação com a máquina de Turing de maneira
   sequencial ou paralela usando concurrent.futures.ProcessPoolExecutor.
3. Executar alguns casos de teste quando rodado como script.

Execute python Main.py para ver a demonstração e o benchmark.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing, time, random
import TuringMachine as tm_module  # importa o módulo completo para manter referência atualizada

def rand_bin(n: int) -> str:
    """Gera uma string binária aleatória.

Args:
    n: Quantidade de bits.

Returns:
    Uma string composta exclusivamente por '0' e '1' de tamanho `n`.
"""
    return ''.join(random.choice("01") for _ in range(n))

def gerar_pares(n: int, k: int):
    """Gera pares de números binários aleatórios.

Args:
    n: Número de pares a gerar.
    k: Quantidade de bits de cada operando.

Yields:
    Tuplas `(a, b)` em que `a` e `b` são strings binárias de tamanho `k`.
"""
    for _ in range(n):
        yield rand_bin(k), rand_bin(k)

def benchmark(pairs, processes: int | None = None) -> float:
    """Mede o tempo de execução para multiplicar diversas tuplas.

Args:
    pairs: Iterável de tuplas `(a, b)` com operandos binários.
    processes: Quantidade de processos a usar. `1` força execução
        sequencial; `None` adota o padrão do sistema; qualquer outro valor
        define o número de workers para um pool.

Returns:
    Tempo total decorrido em segundos.
"""
    start = time.perf_counter()

    if processes == 1:
        for pair in pairs:
            tm_module.simulate(pair)
    else:
        pairs = list(pairs)  # Garante que pode ser percorrido mais de uma vez
        with ProcessPoolExecutor(max_workers=processes) as pool:
            list(pool.map(tm_module.simulate, pairs))

    return time.perf_counter() - start

if _name_ == "_main_":
    multiprocessing.freeze_support()

    # Defina aqui suas multiplicações em binário
    test_cases = [
        ("10", "11"),      # 2 * 3
        ("101", "110"),    # 5 * 6
        ("111", "101"),    # 7 * 5
    ]

    print("Executando casos de teste...")
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(tm_module.simulate, pair) for pair in test_cases]
        for future in as_completed(futures):
            input1, input2, result = future.result()
            expected = bin(int(input1, 2) * int(input2, 2))[2:]
            ok = result == expected
            print(f"{input1} * {input2} => {result} (esperado={expected}) -> {'OK' if ok else 'ERRO'}")

    print("\n--- Benchmark de desempenho ---")
    # Exemplo de benchmark: 10.000 pares de números binários de 5 bits
    num_pairs = 10000
    num_bits = 5
    generated_pairs = list(gerar_pares(num_pairs, num_bits))

    print(f"Benchmark com {num_pairs} pares de números binários de {num_bits} bits:")

    # Benchmark sequencial
    print("Execução sequencial (1 processo):")
    time_sequential = benchmark(generated_pairs, processes=1)
    print(f"Tempo total (sequencial): {time_sequential:.4f} segundos")

    # Benchmark paralelo (usando o número padrão de processos)
    print("Execução paralela (processos padrão do sistema):")
    time_parallel = benchmark(generated_pairs)
    print(f"Tempo total (paralelo): {time_parallel:.4f} segundos")

    print(f"\nGanho de desempenho: {time_sequential / time_parallel:.2f}x mais rápido (paralelo vs sequencial)")