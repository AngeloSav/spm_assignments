#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <hpc_helpers.hpp>
#include <avx_mathfun.h>

const size_t FLOATS_PER_LINE = 8;
const float INFTY = std::numeric_limits<float>::infinity();

float inline find_max(const float *input, size_t K)
{

	v8sf max_line = _mm256_set1_ps(-INFTY);
	size_t i = 0;
	for (; i + FLOATS_PER_LINE - 1 < K; i += FLOATS_PER_LINE)
	{
		v8sf cur_line = _mm256_loadu_ps(&input[i]);
		max_line = _mm256_max_ps(max_line, cur_line);
	}

	float max_val = -INFTY;

	float max_vals[FLOATS_PER_LINE] __attribute__((aligned(32)));
	_mm256_store_ps(max_vals, max_line);
	for (size_t j = 0; j < FLOATS_PER_LINE; j++)
	{
		max_val = std::max(max_val, max_vals[j]);
	}

	// find max of the remainder
	for (; i < K; i++)
	{
		max_val = std::max(max_val, input[i]);
	}

	return max_val;
}

void softmax_avx(const float *input, float *output, size_t K)
{
	float max_val = find_max(input, K);

	// ---------- COMPUTE EXPONENTIALS ------------
	size_t i = 0;
	v8sf sum_line = _mm256_set1_ps(0);
	for (; i + FLOATS_PER_LINE - 1 < K; i += FLOATS_PER_LINE)
	{
		v8sf cur_line = _mm256_loadu_ps(&input[i]);

		cur_line = _mm256_sub_ps(cur_line, _mm256_set1_ps(max_val));

		// compute exponentials
		cur_line = exp256_ps(cur_line);

		// write result in output
		_mm256_storeu_ps(&output[i], cur_line);

		// add to sum
		sum_line = _mm256_add_ps(sum_line, cur_line);
	}

	float sum = 0;

	// reduce sum
	float to_reduce[FLOATS_PER_LINE] __attribute__((aligned(32)));
	_mm256_store_ps(to_reduce, sum_line);
	for (size_t j = 0; j < FLOATS_PER_LINE; j++)
	{
		sum += to_reduce[j];
	}

	// compute output of the remainder
	for (; i < K; i++)
	{
		output[i] = std::exp(input[i]);
		sum += output[i];
	}

	// ---------- NORMALIZE ------------

	i = 0;
	v8sf divisor_line = _mm256_set1_ps(sum);
	for (; i + FLOATS_PER_LINE - 1 < K; i += FLOATS_PER_LINE)
	{
		v8sf cur_line = _mm256_loadu_ps(&output[i]);

		// normalize
		cur_line = _mm256_div_ps(cur_line, divisor_line);

		// write result in output
		_mm256_storeu_ps(&output[i], cur_line);
	}

	// compute output of the remainder
	for (; i < K; i++)
	{
		output[i] /= sum;
	}
}

std::vector<float> generate_random_input(size_t K, float min = -1.0f, float max = 1.0f)
{
	std::vector<float> input(K);
	std::random_device rd;
	std::mt19937 gen(rd());
	// std::mt19937 gen(5489); // fixed seed for reproducible results
	std::uniform_real_distribution<float> dis(min, max);
	for (size_t i = 0; i < K; ++i)
	{
		input[i] = dis(gen);
	}
	return input;
}

void printResult(std::vector<float> &v, size_t K)
{
	for (size_t i = 0; i < K; ++i)
	{
		std::fprintf(stderr, "%f\n", v[i]);
	}
}

int main(int argc, char *argv[])
{
	if (argc == 1)
	{
		std::printf("use: %s K [1]\n", argv[0]);
		return 0;
	}
	size_t K = 0;
	if (argc >= 2)
	{
		K = std::stol(argv[1]);
	}
	bool print = false;
	if (argc == 3)
	{
		print = true;
	}
	std::vector<float> input = generate_random_input(K);
	std::vector<float> output(K);

	TIMERSTART(softime_avx);
	softmax_avx(input.data(), output.data(), K);
	TIMERSTOP(softime_avx);

	// print the results on the standard output
	if (print)
	{
		printResult(output, K);
	}
}
