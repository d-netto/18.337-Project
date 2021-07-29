const DEFAULT_NUM_POLYS = 4

struct StochGalerkinODE{odeFuncType,dimType,varsType,numPolysType,indsDegType,polysType,gridType,normMapType}
  ode_func!::odeFuncType
  dim_::dimType
  vars::varsType
  num_polys::numPolysType
  indices_polys::indsDegType
  poly_basis::polysType
  grid::gridType
  normalization_map::normMapType

  function StochGalerkinODE(ode_func!,dim_,vars;wType=Float64,num_polys=nothing,max_deg=nothing)

    if num_polys === nothing
      num_polys = Tuple(DEFAULT_NUM_POLYS for i in Base.OneTo(length(vars)))
    end

    @assert length(vars) == length(num_polys)

    indices_polys_temp = []
    for index in CartesianIndices(num_polys)
      if max_deg === nothing || mapreduce(i -> index[i] - 1,+,Base.OneTo(length(index))) <= max_deg
        push!(indices_polys_temp,index)
      end
    end
    indices_polys = Tuple(indices_polys_temp)

    num_vars = length(vars)

    poly_basis_temp = Vector{OrthogonalPolyBasis}(undef,num_vars)

    for i in Base.OneTo(length(vars))
      var = vars[i]
      if var isa Uniform
        @assert var.a ≈ -1.0 && var.b ≈ 1.0
        poly_basis_temp[i] = LegendrePolyBasis(num_polys[i];wType)
      elseif var isa Normal
        ## Using physicist's hermite polynomial
        @assert var.μ ≈ 0.0 && var.σ ≈ sqrt(2.0) / 2
        poly_basis_temp[i] = HermitePolyBasis(num_polys[i];wType)
      else
        throw("Not implemented")
      end
    end

    poly_basis = Tuple(poly_basis_temp)

    grid = TensorIntegrationGrid(dim_,num_polys,vars;wType)

    normalization_map = zeros(wType,num_polys)

    for index in CartesianIndices(normalization_map)
        normalization_map[index] = mapreduce(i -> dot(grid.quads[i].weights, (x -> poly_basis[i](x,index[i]-1)^2).(grid.quads[i].nodes)),*,1:length(index))
    end

    return new{typeof(ode_func!),typeof(dim_),typeof(vars),typeof(num_polys),typeof(indices_polys),typeof(poly_basis),
              typeof(grid),typeof(normalization_map)}(ode_func!,dim_,vars,
              num_polys,indices_polys,poly_basis,grid,normalization_map)
  end
end

struct StochGalerkinODECache{yHatType,yHatGridType,intEstType,argCacheType,odeCacheType}
  y_hat::yHatType
  y_hat_for_grid::yHatGridType
  integral_est::intEstType
  arg_cache::argCacheType
  ode_cache::odeCacheType

  function StochGalerkinODECache(s::StochGalerkinODE,u)
    @unpack dim_,vars,num_polys = s

    uType = eltype(u)
    num_vars = length(vars)

    y_hat = zeros(uType,dim_)
    y_hat_for_grid = zeros(uType,dim_,num_polys...)
    integral_est = zeros(uType,dim_)
    arg_cache = zeros(uType,num_vars)
    ode_cache = zeros(uType,dim_)

    return new{typeof(y_hat),typeof(y_hat_for_grid),typeof(integral_est),typeof(arg_cache),typeof(ode_cache)}(
              y_hat,y_hat_for_grid,integral_est,arg_cache,ode_cache)
    end
end

@inline function compute_poly_prod(s::StochGalerkinODE,c::StochGalerkinODECache,index::CartesianIndex)
  @unpack poly_basis = s
  @unpack arg_cache = c
  prod = one(first(arg_cache))
  @inbounds for i in Base.OneTo(length(index))
    prod *= poly_basis[i](arg_cache[i],index[i] - 1)
  end
  return prod
end

@inline function compute_weights_prod(s::StochGalerkinODE,index::CartesianIndex)
  @unpack grid = s
  @unpack quads = grid
  prod = one(eltype(first(quads).weights))
  @inbounds for i in Base.OneTo(length(index))
    prod *= quads[i].weights[index[i]]
  end
  return prod
end

@inline function set_arg!(s::StochGalerkinODE,c::StochGalerkinODECache,index::CartesianIndex)
  @unpack grid = s
  @unpack arg_cache = c
  @unpack quads = grid
  @inbounds for i in Base.OneTo(length(index))
    arg_cache[i] = quads[i].nodes[index[i]]
  end
  nothing
end

function y_hat_expand!(s::StochGalerkinODE,c::StochGalerkinODECache,u)
  @unpack indices_polys,poly_basis = s
  @unpack y_hat,arg_cache = c
  yType = eltype(y_hat)
  fill!(y_hat,zero(yType))
  @inbounds for index in indices_polys
    # Compiler can't infer without this
    prod::yType = compute_poly_prod(s,c,index)
    y_hat .+= (@view u[:,index]) .* prod
  end
  nothing
end

function set_inner_prod!(s::StochGalerkinODE,c::StochGalerkinODECache,u)
  @unpack grid = s
  @unpack y_hat,y_hat_for_grid = c
  @unpack quad_indices = grid
  @inbounds for index in quad_indices
    set_arg!(s,c,index)
    y_hat_expand!(s,c,u)
    y_hat_for_grid[:,index] .= y_hat
  end
  nothing
end

function integrate!(s::StochGalerkinODE,c::StochGalerkinODECache,u,p,t,index::CartesianIndex)
  @unpack ode_func!,poly_basis,grid = s
  @unpack y_hat,y_hat_for_grid,integral_est,arg_cache,ode_cache = c
  @unpack quad_indices = grid
  fill!(integral_est,zero(eltype(integral_est)))
  @inbounds for index_ in quad_indices
    set_arg!(s,c,index_)
    prod = compute_poly_prod(s,c,index) * compute_weights_prod(s,index_)
    y_hat .= (@view y_hat_for_grid[:,index_])
    ode_func!(ode_cache,y_hat,p,arg_cache,t)
    integral_est .+= ode_cache .* prod
  end
  nothing
end

function stoch_galerkin_ode_dynamics!(s::StochGalerkinODE,du,u,p,t)
  c = StochGalerkinODECache(s,u)
  stoch_galerkin_ode_dynamics!(s,c,du,p,t)
  nothing
end

function stoch_galerkin_ode_dynamics!(s::StochGalerkinODE,c::StochGalerkinODECache,du,u,p,t)
  @unpack indices_polys,normalization_map = s
  @unpack integral_est = c
  set_inner_prod!(s,c,u)
  @inbounds for index in indices_polys
    norm_const = normalization_map[index]
    integrate!(s,c,u,p,t,index)
    du[:,index] .= integral_est ./ norm_const
  end
  nothing
end

function (s::StochGalerkinODE)(u0,tspan,p;re_use_cache=true,kwargs...)
  @unpack dim_,num_polys = s
  u_orth_0 = zeros(eltype(u0),dim_,num_polys...)
  u_orth_0[:,CartesianIndex(Tuple(1 for i in num_polys))] .= u0
  if re_use_cache
    local c = StochGalerkinODECache(s,u_orth_0)
    dynamics_iip(du,u,p,t) = stoch_galerkin_ode_dynamics!(s,c,du,u,p,t)
    prob = ODEProblem(dynamics_iip,u_orth_0,tspan,p)
    sol = solve(prob;kwargs...)
  else
    dynamics_oop(du,u,p,t) = stoch_galerkin_ode_dynamics!(s,du,u,p,t)
    prob = ODEProblem(dynamics_oop,u_orth_0,tspan,p)
    sol = solve(prob;kwargs...)
  end
  return sol
end
