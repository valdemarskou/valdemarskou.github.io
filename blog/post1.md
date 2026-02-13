+++
title = "Matrix reductions for eliminating spurious eigenvalues in the Chebyshev Tau method"
hascode = true
hasmath = true
rss = "A short description of the page which would serve as **blurb** in a `RSS` feed; you can use basic markdown here but the whole description string must be a single line (not a multiline string). Like this one for instance. Keep in mind that styling is minimal in RSS so for instance don't expect maths or fancy styling to work; images should be ok though: ![](https://upload.wikimedia.org/wikipedia/en/b/b0/Rick_and_Morty_characters.jpg)"
rss_title = "More goodies"
rss_pubdate = Date(2019, 5, 1)

tags = ["syntax", "code", "image"]

+++



The Cheyshev Tau method is a powerful tool for the numerical solution of eigenvalue boundary value problems. However, when applied to problems of order $\geq 2$, spurious eigenvalues may appear. Largely following \cite{gardner88}, we study and implement a matrix factorization technique that eliminates the spurious values, while still preserving the desired spectral convergence of the resulting numerical scheme. We apply the technique to several examples; notably one where the eigenvalue is located in the boundary. The code developed for this post is found in the following [spectral methods repository](https://github.com/valdemarskou/Julia-spectralmethod).

## Chebyshev approximation

Some background on Chebyshev approxiation is required (see \cite{johnson96} for further details). The Chebyshev polynomial of order $k$ is given by
$$
\begin{align*}
T_k(x) = \arccos(\cos(x)),
\end{align*}
\notag
$$

for $x\in [-1,1]$, and satisfy the three-term recurrence relation $T_{k+1}(x) = 2x T_k (x) - T_{k-1}(x)$, with $T_0(x)=1$ and $T_1(x) = x$. The family $(T_k)_{k\geq 0}$ are an orthogonal basis with respect to the inner product
$$
\langle f,g  \rangle = \int_{-1}^{1} f(x)g(x)(1-x^2)^{-1/2}\text{dx}.
$$

In particular we have $\langle T_i, T_j \rangle = c_i \pi\delta_{ij}/2$, where $c_i=1+\delta_{0i}$. Any smooth function $u$ and its derivatives can therefore be expanded as modal Chebyshev series:

$$
\begin{align*}
u(x) = \sum_{n=0}^\infty a_n T_n(x),  && u^{(m)}(x) = \sum_{n=0}^\infty a_n^{(m)} T_n(x), && m=1,2,\ldots
\end{align*}
$$

Formulas for the derivative coefficients can be derived from the recursion relation $c_{n-1}a_{n-1}^{(m)} = 2na_n^{(m-1)} + a_{n+1}^{(m)}$, $n\geq 1$. In particular we have:
$$
\begin{aligned}
c_n a_n^{(1)} &= 2\sum_{\substack{p=n+1 \\ p + n\text{ odd}}}^\infty p a_p, \\
c_n a_n^{(2)} &= \sum_{\substack{p=n+2 \\ p + n\text{ even}}}^\infty p(p^2 -n^2)a_p, \\
c_n a_n^{(3)} &= \frac{1}{4}\sum_{\substack{p=n+3 \\ p + n\text{ odd}}}^\infty p(p^2(p^2-2) -2p^2n^2+(n^2-1))a_p, \\
c_n a_n^{(4)} &= \frac{1}{24}\sum_{\substack{p=n+4 \\ p + n\text{ even}}}^\infty p(p^2(p^2-4)^2 - 3p^4n^2 +3p^2n^4 - n^2(n^2-4)^2)a_p
\end{aligned}
$$

Formulas for higher derivatives can likewise be obtained. Another useful result gives the values of the $m$'th order derivative of a Chebyshev polynomial when evaluated at the end points:
$$
T_n^{(m)}(\pm 1) = (\pm 1)^{n+m}\prod_{k=0}^{m-1}\frac{n^2-k^2}{2k+1}. \label{cheb-boundary-identity}
$$

The above formulas are in terms of infinite sums, but in practice, we will work with truncated (finite) modal expansions. 


## Fourth order equations and the modified Tau method

We first consider a general fourth order equation of the form: 
$$
\begin{cases}
L_1 v + L_2 u - s(v+L_3 u)=0 & x\in\Omega \\
v = L_4 u
\end{cases}
$$

where the $L_i$'s are (linear) second order differential operators, $u,v$ are the unknown eigenfunctions defined on a domain $\Omega$, and $s$ is the unknown eigenvalue. The problem may of course be formulated exclusively in terms of the eigenfunction $u$, but the factorization will be useful later. The Tau method involves truncating the Chebyshev expansions of $u$ and $v$ and taking the inner product to obtain a matrix system of equations. Unique to the Tau method is that the boudary conditions are imposed exactly, using \eqref{cheb-boundary-identity}. For our modified method, we write:
$$
\begin{align*}
u(x) \approx\sum_{n=0}^{N_{int}+N_{bou}-2} a_n T_n(x),  && v(x) \approx \sum_{n=0}^{N_{int}+N_{bou}-2} b_n T_n(x),
\end{align*}
$$

where $N=N_{int}$ and $N_{bou}$ signifies the number of equations we obtain from the interior (governing) equations, and boundary conditions, respectively. Since we consider fourth order equations, we need exactly four boundary conditions, $N_{bou} = 4$, for the equation to be well-posed. Substituting the approximations into the governing equations and taking the inner product, we obtain a matrix system of the form:
$$
\begin{aligned}
&\bm{b} = Q\bm{a} \\
&B\bm{b} + A\bm{a} -s(\bm{b}+C\bm{a})=(\bm{0},\bm\tau)^T,
\end{aligned}
$$

where the $A,B,C,Q\in\R^{(N+3)^2}$ are the coefficient operators derived from $L_2,L_1,L_3,L_4$, respectively. The $\bm\tau=(\tau_1,\tau_2)$ coefficients are unknowns, and are used to indicate the rate of convergence. We now write $\bm{b}_1 = (b_0,\ldots,b_{N-2})^T$, $\bm{b}_2 = (b_{N-1},b_N)^T$, $\bm{y} = (b_{N+1},b_{N+2})$, and partition the matrices according to the following schematic:
$$
\begin{aligned}
  &\left( 
    \begin{array}{ccc|c}
      \\
      & B_1 & & B_4 \\
      \\
      \hline
      & B_2 & & B_5 \\
      \hline
      & B_3 & & B_6
    \end{array}
  \right)

\left(\begin{array}{c}
    \\
    \bm{b}_1 \\
    \\
    \hline
    \bm{b}_2 \\
    \hline
    \bm{y}
  \end{array}\right)

+
\left(\begin{array}{c}
  \\
  A_1 \\
  \\
  \hline
  A_2 \\
  \hline
  A_3
\end{array}\right)

\left(\begin{array}{c}
  \\
  \\
  \bm{a} \\
  \\
  \\
\end{array}\right)
\\
&\phantom{--------}-s\left(\begin{array}{c}
  \\
    \bm{b}_1 \\
    \\
    \hline
    \bm{b}_2 \\
    \hline
    \bm{y}
\end{array}\right)

-s\left(\begin{array}{c}
  \\
  C_1 \\
  \\
  \hline
  C_2 \\
  \hline
  C_3
\end{array}\right)

\left(\begin{array}{c}
  \\
  \\
  \bm{a} \\
  \\
  \\
\end{array}\right)

= 
\left(\begin{array}{c}
  \\
  \\
  \bm{0}
  \\
  \\
  \hline
  \bm\tau
\end{array}\right)

\\
\\
&\left(\begin{array}{c}
  \\
    \bm{b}_1 \\
    \\
    \hline
    \bm{b}_2 \\
    \hline
    \bm{y}
\end{array}\right) = \left(\begin{array}{c}
  \\
  Q_1 \\
  \\
  \hline
  Q_2 \\
  \hline
  Q_3
\end{array}\right)

\left(\begin{array}{c}
  \\
  \\
  \bm{a} \\
  \\
  \\
\end{array}\right)
\end{aligned}
$$

Writing $\bm{\hat{b}}$, we expand the block matrix equations:
$$
\begin{aligned}
  & B_1\bm{\hat{b}} + B_4\bm{y} + A_1\bm{a} - s(\bm{b}_1+C_1\bm{a})=0, \\
  & B_2\bm{\hat{b}} + B_5\bm{y} + A_2\bm{a} - s(\bm{b}_2+C_2\bm{a})=0.
\end{aligned}
$$

We can then isolate $\bm{y}$ in the second equation, requiring only a matrix inversion of $B_5\in\R^{2\times 2}$. With $\bm{b}_1=Q_1\bm{a}$ and $\bm{b}_2 = Q_2\bm{a}$ we get:
$$
\begin{aligned}
  \bm{y} &=-B_5^{-1}\left[B_2\bm{b}+A_2\bm{a}-s(\bm{b}_2+C_2\bm{a}) \right] \\
  &= -B_5^{-1}[B_2 Q + A_2 -s(Q_2 + C_2)]\bm{a}
\end{aligned}
$$

Substituting this into the first equation yields
$$
\begin{aligned}
  [B_1Q-B_4 B_5^{-1}B_2Q + A_1 - B_4 B_5^{-1}A_2]\bm{a} = s[Q_1 + C_1 - B_4B_5^{-1}(Q_2 + C_2)]\bm{a},
\end{aligned}
$$

which defines a generalized eigenvalue problem of the form $M\bm{a} = sN\bm{a}$, where $M,N\in\R^{(N-1)\times(N+3)}$. To obtain a square system, the boundary conditions may be viewed as a matrix equation $D\bm{a} =\bm{0}$ (with $D\in\R^{4\times (N+3)}$) that we concatenate to $M$. Writing $\bm{a_1}=(a_0,\ldots,a_N)^T$ amd $\bm{a_2} = (a_{N+1},a_{N+2})^T$ we further partition the complete system:
$$
\begin{aligned}
  \left(\begin{array}{c|c}
    M_1 & M_2 \\
    \hline
    D_1 & D_2
  \end{array}\right)
\end{aligned}

\left(\begin{array}{c}
  \bm{a_1} \\
  \hline
  \bm{a_2}
\end{array}\right)

= 
s
\begin{aligned}
  \left(\begin{array}{c|c}
    N_1 & N_2 \\
    \hline
    0 & 0
  \end{array}\right)
\end{aligned}\bm{a}.
$$

From the second block matrix equation we get $\bm{a_2} = -D_2^{-1} D_1\bm{a_1}$. Substituting into the first block matrix equation yields our final generalized eigenvalue problem:
$$
  (M_1 - M_2 D_2^{-1} D_1)\bm{a_1} = s(N_1 - N_2 D_2^{-1} D_1)\bm{a_1},
$$

which can be solved with any appropriate eigenvalue solver. The final coefficients $\bm{a_2}$ are recovered from the previous equation. \\

The tau coefficients $\bm{\tau} = (\tau_1,\tau_2)^T$ are computed from the equation:
$$
  B_3\bm{\hat{b}} + B_6\bm{y} + A_3\bm{a} - s(\bm{y}+C_3\bm{a})=\bm{\tau},
$$

Its magnitude is useful for identifying spurious eigenvalues, as well as indicating the degree of convergence of the modal coefficients $\bm{a}$. However, computing $\bm{\tau}$ also requires computing a normalized eigenvector corresponding to the eigenvalue, an increase in computational cost. Since only use the magnitude of the $\bm{\tau}$ coefficients are useful, we may simply consider the sum of its absolute values: if any one of the coefficients for a given eigenvalue have large magnitude, then that value is spurious or a poor approximation to a true eigenvalue.

## Examples

**Eigenvalue in the interior equation:** 

Consider the fourth order boundary value problem
$$
\begin{aligned}
  \begin{cases}
    u'''' + Ru'''-su''=0 & x\in [-1,1] \\
    u(\pm 1) = u'(\pm 1) =0
  \end{cases}
\end{aligned}
$$

where $R$ is a real parameter and $s$ is the eigenvalue. The following code solves the problem using the modified Tau method: \\
\\

\collaps{Click to expand/retract code}{
```julia
B = chebyshevSecondDerivativeMatrix(N_int+2,0) + R*chebyshevFirstDerivativeMatrix(N_int+2,0)
Q = chebyshevSecondDerivativeMatrix(N_int,2)

B1 = @view B[1:N_int-1,1:N_int+1]
B2 = @view B[N_int:N_int+1,1:N_int+1]
B4 = @view B[1:N_int-1,N_int+2:N_int+3]
B5 = @view B[N_int:N_int+1,N_int+2:N_int+3]

B3 = @view B[N_int+2:N_int+3,1:N_int+1]
B6 = @view B[N_int+2:N_int+3,N_int+2:N_int+3]

Q1 = @view Q[1:N_int-1,:]
Q2 = @view Q[N_int:N_int+1,:]

B5inv = inv(B5)

M = B1*Q-B4*B5inv*B2*Q
N = Q1 - B4*B5inv*Q2

BC = zeros(4,N_int+3)
BC[1, :] .= (-1).^(0:N_int+2)
BC[2, :] .= 1
BC[3, :] .= -(-1).^(0:N_int+2) .* (0:N_int+2).^2
BC[4, :] .= (0:N_int+2).^2

M1 = @view M[1:N_int-1,1:N_int-1]
M2 = @view M[1:N_int-1,N_int:N_int+3]
M3 = @view BC[1:4,1:N_int-1]
M4 = @view BC[1:4,N_int:N_int+3]

N1 = @view N[1:N_int-1,1:N_int-1]
N2 = @view N[1:N_int-1,N_int:N_int+3]

M4inv = inv(M4)

values,vectors = eigen(M1-M2*M4inv*M3,N1-N2*M4inv*M3)

vectors = [
vectors;
-M4inv*M3 *vectors
]

# Normalize + sign
vectors .= vectors ./ sum(abs, vectors; dims=1)
vectors .*= reshape(sign.(vectors[1, :]), 1, :)


# Tau values
y = -B5inv*B2*Q*vectors + B5inv*Q2*vectors*diagm(values)
tau = B3*Q*vectors + B6*y - y*diagm(values)
tau = sum(abs, tau; dims=1)

return values,vectors,tau
```
}




### Example 2 - eigenvalue in the boundary condition:





### References

* \biblabel{gardner88}{Gardner et al.} **Gardner**, **Trogdon** and **Douglas**, [A Modified Tau Spectral Method That Eliminates Spurious Eigenvalues](https://www.sciencedirect.com/science/article/abs/pii/0021999189900934), 1988.
* \biblabel{johnson96}{Johnson} **Duane Johnson**, [Chebyshev Polynomials in the Spectral Tau Method and Applications to Eigenvalue Problems](https://ntrs.nasa.gov/api/citations/19960029104/downloads/19960029104.pdf), 1996.
<!--

# More goodies

\toc

## More markdown support

The Julia Markdown parser in Julia's stdlib is not exactly complete and Franklin strives to bring useful extensions that are either defined in standard specs such as Common Mark or that just seem like useful extensions.

* indirect references for instance [like so]

[like so]: http://existentialcomics.com/

or also for images

![][some image]

some people find that useful as it allows referring multiple times to the same link for instance.

[some image]: https://upload.wikimedia.org/wikipedia/commons/9/90/Krul.svg

* un-qualified code blocks are allowed and are julia by default, indented code blocks are not supported by default (and there support will disappear completely in later version)

```
a = 1
b = a+1
```

you can specify the default language with `@def lang = "julia"`.
If you actually want a "plain" code block, qualify it as `plaintext` like

```plaintext
so this is plain-text stuff.
```

## A bit more highlighting

Extension of highlighting for `pkg` an `shell` mode in Julia:

```julia-repl
(v1.4) pkg> add Franklin
shell> blah
julia> 1+1
(Sandbox) pkg> resolve
```

you can tune the colouring in the CSS etc via the following classes:

* `.hljs-meta` (for `julia>`)
* `.hljs-metas` (for `shell>`)
* `.hljs-metap` (for `...pkg>`)

## More customisation

Franklin, by design, gives you a lot of flexibility to define how you want stuff be done, this includes doing your own parsing/processing and your own HTML generation using Julia code.

In order to do this, you can define two types of functions in a `utils.jl` file which will complement your `config.md` file:

* `hfun_*` allow you to plug custom-generated HTML somewhere
* `lx_*` allow you to do custom parsing of markdown and generation of HTML

The former (`hfun_*`) is most likely to be useful.

### Custom "hfun"

If you define a function `hfun_bar` in the `utils.jl` then you have access to a new template function `{{bar ...}}`. The parameters are passed as a list of strings, for instance variable names but it  could just be strings as well.

For instance:

```julia
function hfun_bar(vname)
  val = Meta.parse(vname[1])
  return round(sqrt(val), digits=2)
end
```

~~~
.hf {background-color:black;color:white;font-weight:bold;}
~~~

Can be called with `{{bar 4}}`: **{{bar 4}}**.

Usually you will want to pass variable name (either local or global) and collect their value via one of `locvar`, `globvar` or `pagevar` depending on your use case.
Let's have another toy example:

```julia
function hfun_m1fill(vname)
  var = vname[1]
  return pagevar("menu1", var)
end
```

Which you can use like this `{{m1fill title}}`: **{{m1fill title}}**. Of course  in this specific case you could also have used `{{fill title menu1}}`: **{{fill title menu1}}**.

Of course these examples are not very useful, in practice you might want to use it to generate actual HTML in a specific way using Julia code.
For instance you can use it to customise how [tag pages look like](/menu3/#customising_tag_pages).

A nice example of what you can do is in the [SymbolicUtils.jl manual](https://juliasymbolics.github.io/SymbolicUtils.jl/api/) where they use a `hfun_` to generate HTML encapsulating the content of code docstrings, in a way doing something similar to what Documenter does. See [how they defined it](https://github.com/JuliaSymbolics/SymbolicUtils.jl/blob/website/utils.jl).

**Note**: the  output **will not** be reprocessed by Franklin, if you want to generate markdown which should be processed by Franklin, then use `return fd2html(markdown, internal=true)` at the end.

### Custom "lx"

These commands will look the same as latex commands but what they do with their content is now entirely controlled by your code.
You can use this to do your own parsing of specific chunks of your content if you so desire.

The definition of `lx_*` commands **must** look like this:

```julia
function lx_baz(com, _)
  # keep this first line
  brace_content = Franklin.content(com.braces[1]) # input string
  # do whatever you want here
  return uppercase(brace_content)
end
```

You can call the above with `\baz{some string}`: \baz{some string}.

**Note**: the output **will be** reprocessed by Franklin, if you want to avoid this, then escape the output by using `return "~~~" * s * "~~~"` and it will be plugged  in as is in the HTML.

-->
