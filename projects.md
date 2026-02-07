+++
title = "Academic projects"
hascode = true
date = Date(2019, 3, 22)
rss = "A short description of the page which would serve as **blurb** in a `RSS` feed; you can use basic markdown here but the whole description string must be a single line (not a multiline string). Like this one for instance. Keep in mind that styling is minimal in RSS so for instance don't expect maths or fancy styling to work; images should be ok though: ![](https://upload.wikimedia.org/wikipedia/en/3/32/Rick_and_Morty_opening_credits.jpeg)"

tags = ["syntax", "code"]
+++

## MSc thesis.

I defended my MSc thesis "Approximating solutions to partial differential equations with physics-informed reproducing kernels" and earned the degree of Master of Science in Mathematics on July 21st, 2025. You can download the~~~
<a href="/assets/Msc_thesis_report.pdf">thesis,</a>
~~~
the~~~
<a href="/assets/Msc_thesis_slides.pdf">defense slides,</a>
~~~
or read the abstract below. The code for the project is found on my main GitHub page.

### Abstract of the thesis:
"In this thesis, we present and explore the reproducing kernel Hilbert space regression framework in the context of estimating behavior governed by physics-based partial differential equations. We implement the framework with different optimization procedures and equation examples, and evaluate our results based on how we the underlying equation behavior is captured. We present a method for transforming the reproducing kernel to naturally uphold global properties imposed by such equations, in the form of conservation laws, and demonstrate that the resulting kernel expansion estimators are more physically valid and also yield improved numerical accuracy."

## BSc thesis.
I defended my BSc thesis "Qualitative Aspects of Mathematical Reaction Networks
Using Methods from Algebra and Degree Theory" and earned the degree of Bachelor of Science in Mathematics on September 8th, 2022. You can download the~~~
<a href="/assets/Bachelorprojekt_Final.pdf">thesis</a>
~~~
or read the abstract below.

### Abstract of the thesis:
"The goal of this project is to give an introduction some areas within chemical reaction network theory. The structure of reaction networks and its interplay with the associated systems of differential equations is described, and different methods using tools from both algebra and analysis are introduced and applied to several concrete networks. A large part of the project is dedicated to a given network’s capacity for multistationarity, which is an interesting property concerning the number of steady states for the associated ODE system. The method developed here involves computing the $\mathcal{C}^1$-mapping degree for a certain family of functions $\phi_c\in\mathcal{C}^1(\R^n_{\geq 0},\R^n)$. Since many of the properties of the mapping degree are nontrivial, their proofs are given in full detail. Finally, the project is wrapped up by applying all of the methods developed on several specific reaction networks."

## Project in nonlinear elliptic PDE.
I wrote a project outside course scope in regularity for nonlinear elliptic partial differential equations. Essentially it was a reading course in the book "Fully Nonlinear Elliptic Equations" by Caffarelli and Cabré, where I focused on the details of powerful covering arguments that go unmentioned in standard textbooks, as well as the details of the Krylov-Safonov Harnack inequlity. You can download the ~~~
<a href="/assets/Bachelorprojekt_Final.pdf">report</a>
~~~
or read the abstract below.

### Abstract of the project:
"The purpose of this report is to establish Hölder regularity results for some fully nonlinear elliptic PDE, of second order. For this purpose, the viscosity solution is an essential construction, and we briefly introduce the appropriate function spaces that arise in this setting. We show the Harnack inequality and the Evans-Krylov theorem in this setting, and then we move on to solving the Dirichlet problem for concave equations with constant coefficients."


## Project in spectral element method for 2D Navier Stokes.
I wrote a small project focusing on developing a solver for the 2D Navier Stokes equation in simple periodic domains. You can download the resulting ~~~
<a href="/assets/Numerical_Methods_assignment03_poster.pdf">poster</a>
~~~
or check out the [code repository](https://github.com/valdemarskou/Numerical-Methods-for-DE) on GitHub.


## Project in trace inequalities and Stahls theorem.
I wrote a small project about trace inequalities, where I gave a shortened proof of Stahl's theorem, which represents a class of Hermitian matrix trace formulae in terms of the Laplace transform of a positive measure. A number of interesting corollaries was also dicussed, in particular the connections with Bernstein's theorem for monotone functions. The project was mostly of theoretical interest, but but also has applications in combining data-driven methods with computational linear algebra. You can download the report~~~
<a href="/assets/Msc_Project_2.pdf">here.</a>
~~~

<!--

# Working with code blocks

\toc

## Live evaluation of code blocks

If you would like to show code as well as what the code outputs, you only need to specify where the script corresponding to the code block will be saved.

Indeed, what happens is that the code block gets saved as a script which then gets executed.
This also allows for that block to not be re-executed every time you change something _else_ on the page.

Here's a simple example (change values in `a` to see the results being live updated):

```julia:./exdot.jl
using LinearAlgebra
a = [1, 2, 3, 3, 4, 5, 2, 2]
@show dot(a, a)
println(dot(a, a))
```

You can now show what this would look like:

\output{./exdot.jl}

**Notes**:
* you don't have to specify the `.jl` (see below),
* you do need to explicitly use print statements or `@show` for things to show, so just leaving a variable at the end like you would in the REPL will show nothing,
* only Julia code blocks are supported at the moment, there may be a support for scripting languages like `R` or `python` in the future,
* the way you specify the path is important; see [the docs](https://franklinjl.org/code/#more_on_paths) for more info. If you don't care about how things are structured in your `/assets/` folder, just use `./scriptname.jl`. If you want things to be grouped, use `./group/scriptname.jl`. For more involved uses, see the docs.

Lastly, it's important to realise that if you don't change the content of the code, then that code will only be executed _once_ even if you make multiple changes to the text around it.

Here's another example,

```julia:./code/ex2
for i ∈ 1:5, j ∈ 1:5
    print(" ", rpad("*"^i,5), lpad("*"^(6-i),5), j==5 ? "\n" : " "^4)
end
```

which gives the (utterly useless):

\output{./code/ex2}

note the absence of `.jl`, it's inferred.

You can also hide lines (that will be executed nonetheless):

```julia:./code/ex3
using Random
Random.seed!(1) # hide
@show randn(2)
```

\output{./code/ex3}


## Including scripts

Another approach is to include the content of a script that has already been executed.
This can be an alternative to the description above if you'd like to only run the code once because it's particularly slow or because it's not Julia code.
For this you can use the `\input` command specifying which language it should be tagged as:


\input{julia}{/_assets/scripts/script1.jl} 


these scripts can be run in such a way that their output is also saved to file, see `scripts/generate_results.jl` for instance, and you can then also input the results:

\output{/_assets/scripts/script1.jl} 

which is convenient if you're presenting code.

**Note**: paths specification matters, see [the docs](https://franklinjl.org/code/#more_on_paths) for details.

Using this approach with the `generate_results.jl` file also makes sure that all the code on your website works and that all results match the code which makes maintenance easier.
-->
