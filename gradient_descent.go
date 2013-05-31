package gradient_descent


import "math"

// Multiplies `alpha` by every value in `vector2`, then
// adds that result with `vector`
func Scale_add(vector1 []float64, vector2 []float64, alpha float64) []float64 {
	scaled := make([]float64, 0)
	result := make([]float64, 0)

	for _, value := range vector2 {
		scaled = append(scaled, value * alpha)
	}

	for i, value := range scaled {
		scaled = append(result, value + vector1[i])
	}

	return result
}



// the diff function passed to gradient descent
type DifferentiatedFunction struct {
	convergenceThreshold float64
}

func (diff_func *DifferentiatedFunction) f(theta []float64) (result float64) {
	return
}

func (diff_func *DifferentiatedFunction) diff_of_f(theta []float64) (result []float64) {
	return
}

func (diff_func *DifferentiatedFunction) is_converged(vec1, vec2 []float64) bool {
	if diff_func.convergenceThreshold <= 0.0 {
		diff_func.convergenceThreshold = float64(1e-8)
	}
	return math.Abs(diff_func.f(vec1) - diff_func.f(vec2)) < diff_func.convergenceThreshold
}

func Gradient_descent(guess_vector []float64, alpha float64, diff_func DifferentiatedFunction) (outputs []float64)  {
	last_guess := guess_vector

	outputs = Scale_add(last_guess, diff_func.diff_of_f(last_guess), -alpha)

	for {
		if diff_func.is_converged(last_guess, outputs) {
			break
		}

		last_guess = outputs
		outputs = Scale_add(last_guess, diff_func.diff_of_f(last_guess), -alpha)
	}
	return
}
