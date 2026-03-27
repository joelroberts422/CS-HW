## Part 1: Conceptual Questions

1. Tracing based compilation executes the function once with an example, records the operations into a graph, and then compiles based on that graph. Scripting compiles without running the code and uses the source code and logic directly. A python function with an if/else conditional would not be traceable because the logic would only trace one path, but scripting would be able to handle that logic and compile two paths for each case.

2. The for loop in the given function would be an issue as the traced graph would not handle the native python control and know how to compile into fewer kernel launches. In this case, scripting would be the better choice, so using either torch.jit.script or using a jax.lax.fori_loop rollup if the number of loops is set.

3. 
    a. There would be 4 kernel launches for cosine, sine, exp, and the final addition. There would be 5 memory round trips, twice "x" for cosine and exp, once for cos(x), once for sin(cos(x)), and once for exp(x).
    b. There would be one kernel that combines all operations in fused mode and one memory read of "x" at the beginning.
    c. This reduces the theoretical memory bandwidth reduction factor by a factor of 5, from 5 to 1.

4. Dynamic control flow is a challenge for compilation because it branches the computation graph into separate branches. Tracing won't work because not all possible branches of the function will be traversed in the first run. Solutions would be using jax.lax.cond for a tensor based conditional, making the conditional flag static, or for pytorch, using a script based compiler to compile the several branches directly from the logic.

5. torch.compile runs slower in the first step because it must create the computation graph and compile it to machine code directly.