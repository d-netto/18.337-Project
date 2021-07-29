cd("..")
include(joinpath(pwd(), "src/utils.jl"))

const e_ = 10^-2
const degree_1 = 5
const degree_2 = 5
const dim_ = 2
const length_quad_= 6
quad = gausslegendre(length_quad_)
const nodes = quad[1]
const weights = quad[2]

function create_normalization_map(nodes,weights)
  normalization_map = Matrix{Float64}(undef,degree_1,degree_2)
  @inbounds for i in 1:degree_1
    @inbounds for j in 1:degree_2
      normalization_map[i,j] = dot(weights, (x -> Pl(x,i-1)^2).(nodes)) * dot(weights, (x -> Pl(x,j-1)^2).(nodes))
    end
  end
  return normalization_map
end

const norm_map_ = create_normalization_map(nodes,weights)

## Expands the tensor product representation and stores it into alloc_expand_tensor
function tensor_legendre_expand!(alloc_expand_tensor,u,x_1,x_2)
  fill!(alloc_expand_tensor,zero(typeof(first(alloc_expand_tensor))))
  @inbounds for i in 1:degree_1
    pl_1 = Pl(x_1,i-1)
    @inbounds for j in 1:degree_2
      alloc_expand_tensor .+= (@view u[:,i,j]) .* (pl_1 * Pl(x_2,j-1))
    end
  end
  nothing
end

function set_inner_prod!(alloc_expand_tensors,alloc_expand_inner_prod,u)
  @inbounds for i in 1:length_quad_
    x_1 = nodes[i]
    @inbounds for j in 1:length_quad_
      x_2 = nodes[j]
      tensor_legendre_expand!(alloc_expand_tensors,u,x_1,x_2)
      alloc_expand_inner_prod[:,i,j] .= alloc_expand_tensors
    end
  end
  nothing
end

function integrate_!(alloc_expand_integral,alloc_expand_inner_prod,i,j,u,p,t,norm_const)
  fill!(alloc_expand_integral,zero(typeof(first(alloc_expand_integral))))
  @inbounds for i_ in 1:length_quad_
    x_1 = nodes[i_]
    prod = Pl(x_1,i-1) * weights[i_]
    @inbounds for j_ in 1:length_quad_
      x_2 = nodes[j_]
      prod_ = prod * Pl(x_2,j-1) * weights[j_]
      u_1 = alloc_expand_inner_prod[1,i_,j_]
      u_2 = alloc_expand_inner_prod[2,i_,j_]
      alloc_expand_integral[1] += ((p[1] + e_ * x_1) * u_1 + (-p[2] - e_ * x_1 * x_2) * u_1 * u_2) * prod_
      alloc_expand_integral[2] += ((-p[3] - e_ * x_1 * x_2) * u_2 + (p[4] + e_ * x_2) * u_1 * u_2) * prod_
    end
  end
  nothing
end

function lotka_volterra_coeffs_orth_closure(du,u,p,t,alloc_expand_tensors,alloc_expand_integrals,alloc_expand_inner_prod)
  set_inner_prod!(alloc_expand_tensors,alloc_expand_inner_prod,u)
  @inbounds for i in 1:degree
    @inbounds for j in 1:degree
      norm_const = norm_map_[i,j]
      integrate_!(alloc_expand_integrals,alloc_expand_inner_prod,i,j,u,p,t,norm_const)
      du[:,i,j] .= alloc_expand_integrals ./ norm_const
    end
  end
  nothing
end

## Allocate arrays
alloc_expand_tensors = zeros(Float64,dim_)
alloc_expand_integrals = zeros(Float64,dim_)
alloc_expand_inner_prod = zeros(Float64,dim_,length_quad_,length_quad_)

## Set up parameters
u_orth_0 = zeros(dim_,degree_1,degree_2)
u_orth_0[:,1,1] = ones(dim_)
tspan = (0.0,10.0)
p_m = [1.5,1.0,3.0,1.0]

## Solve modified problem
lotka_volterra_coeffs_orth(du,u,p,t) = lotka_volterra_coeffs_orth_closure(du,u,p,t,alloc_expand_tensors::Vector{Float64},alloc_expand_integrals::Vector{Float64},alloc_expand_inner_prod::Array{Float64,3})
prob = ODEProblem(lotka_volterra_coeffs_orth,u_orth_0,tspan,p_m)

sol = solve(prob, AutoTsit5(Rodas5()))

## Check for consistency
interval = 0.0:0.1:10.0
x_1 = rand(Uniform(-1.0, 1.0))
x_2 = rand(Uniform(-1.0, 1.0))
plot(interval, [sum([sol(t)[1,i,j] * Pl(x_1,i-1) * Pl(x_2,j-1) for i in 1:degree for j in 1:degree]) for t in interval])
plot!(interval, [sum([sol(t)[2,i,j] * Pl(x_1,i-1) * Pl(x_2,j-1) for i in 1:degree for j in 1:degree]) for t in interval])
