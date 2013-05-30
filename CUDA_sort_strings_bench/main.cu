#include <iostream>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sequence.h>
#include <thrust/mismatch.h>
#include <cstdlib>
#include <ctime>
#include <climits>

#include "strings_type.h"

#define ROWS_COUNT 1024*1024
//#define ROWS_LEN 8


struct T_swap_le_be_64 {
	__host__ __device__ 
	inline unsigned long long operator()(unsigned unsigned long long const& val) const {
		return ((((unsigned long long)255<<(8*7)) & val) >> (8*7)) |
			((((unsigned long long)255<<(8*6)) & val) >> (8*5)) |
			((((unsigned long long)255<<(8*5)) & val) >> (8*3)) |
			((((unsigned long long)255<<(8*4)) & val) >> (8*1)) |
			((((unsigned long long)255<<(8*3)) & val) << (8*1)) |
			((((unsigned long long)255<<(8*2)) & val) << (8*3)) |
			((((unsigned long long)255<<(8*1)) & val) << (8*5)) |
			((((unsigned long long)255<<(8*0)) & val) << (8*7));
	}
};

struct T_swap_le_be_32 {
	__host__ __device__ 
	inline unsigned long long operator()(unsigned unsigned long long const& val) const {
		return ((val>>24)) |			// move byte 3 to byte 0
				((val<<8)&0xff0000) |	// move byte 1 to byte 2
				((val>>8)&0xff00) |		// move byte 2 to byte 1
				((val<<24));			// byte 0 to byte 3
	}
};

struct T_swap_le_be_16 {
	__host__ __device__ 
	inline unsigned long long operator()(unsigned unsigned long long const& val) const {
		return (val<<8) |		// move byte 0 to byte 1
				(val>>8);		// move byte 1 to byte 0
	}
};


template<typename Str>
struct rand_string {
	__host__ __device__ Str operator()() {
		Str str;
		for(size_t i = 0; i < Str::size; ++i) str.data[i] = rand() % 256;
		return str;
	}
};
// --------------------------------------------------------------------------------


template<size_t string_length >
void test_case(const size_t cardinality = ROWS_COUNT) {
	double t_str1, t_str2, t_str_radix8, t_str3, t_str4;
	std::cout.precision(3);

	/// Timers
	clock_t start, end;
	std::cout << std::endl << "Test case for static strings with string_length = " << string_length << ", cardinality = " << cardinality << std::endl;

	/// generate random strings on host
	thrust::host_vector<Str<string_length> > h_v(ROWS_COUNT);

	const size_t repeates = ROWS_COUNT/cardinality;
	thrust::generate(h_v.begin(), h_v.begin() + cardinality, rand_string<Str<string_length> >());
	for(size_t i = 1; i < repeates; ++i) {
		thrust::copy(h_v.begin(), h_v.begin() + cardinality, h_v.begin() + cardinality*i);
	}

	/// prepare for sequence
	thrust::device_vector<unsigned int> d_indecies(ROWS_COUNT);

	/// sort result for compare
	thrust::host_vector<unsigned int> h_i1, h_i_result;
	thrust::host_vector<Str<string_length> > h_v1;

	{
		typedef typename Str<string_length> T_str;
		/// copy host to device
		thrust::device_vector<T_str> d_v = h_v;

		/// generate sequence indecies
		thrust::sequence(d_indecies.begin(), d_indecies.end());
		cudaThreadSynchronize();

		/// Test time for stable sort by key
		start = clock();
		thrust::stable_sort_by_key(d_v.begin(), d_v.end(), d_indecies.begin());
		cudaThreadSynchronize();
		end = clock();
		t_str1 = static_cast<double>(end-start)/CLOCKS_PER_SEC;
		std::cout << "Str1<" << string_length << "> Elapsed: " << t_str1 << " sec." << std::endl;
		h_i1 = d_indecies;
		h_v1 = d_v;
	}

	{
		typedef typename Str2<string_length> T_str;
		/// copy host to device
		thrust::device_vector<T_str > d_v(ROWS_COUNT);
		thrust::copy((T_str *)h_v.data(), (T_str *)h_v.data() + ROWS_COUNT, d_v.data());

		/// generate sequence indecies
		thrust::sequence(d_indecies.begin(), d_indecies.end());
		cudaThreadSynchronize();

		/// Test time for stable sort by key
		start = clock();
		thrust::stable_sort_by_key(d_v.begin(), d_v.end(), d_indecies.begin());
		cudaThreadSynchronize();
		end = clock();
		t_str2 = static_cast<double>(end-start)/CLOCKS_PER_SEC;
		std::cout << "Str2<" << string_length << "> Elapsed: " << t_str2 << " sec. ";
		std::cout << "Faster than Str1: " << t_str1/t_str2 << " X. ";
		h_i_result = d_indecies;
		std::cout << "Indexes " << (thrust::equal(h_i1.begin(), h_i1.end(), h_i_result.begin())?"equal":"differ");
		thrust::host_vector<T_str > h_v_result = d_v;
		std::cout << ",data " << (thrust::equal(h_v_result.data(), h_v_result.data() + ROWS_COUNT, (T_str *)h_v1.data())?"equal":"differ") << std::endl;
	}

	if(string_length == 8)
	{
		typedef typename Str<8> T_str;
		/// copy host to device
		thrust::device_vector<T_str > d_v(ROWS_COUNT);
		thrust::copy((T_str *)h_v.data(), (T_str *)h_v.data() + ROWS_COUNT, d_v.data());

		/// generate sequence indecies
		thrust::sequence(d_indecies.begin(), d_indecies.end());

		thrust::device_ptr<unsigned long long> ptr( reinterpret_cast<unsigned long long *>(thrust::raw_pointer_cast(d_v.data() )));
		cudaThreadSynchronize();

		/// Test time for stable sort by key
		start = clock();
		thrust::transform(ptr, ptr + ROWS_COUNT, ptr, T_swap_le_be_64());
		thrust::stable_sort_by_key(ptr, ptr + ROWS_COUNT, d_indecies.begin());
		thrust::transform(ptr, ptr + ROWS_COUNT, ptr, T_swap_le_be_64());
		cudaThreadSynchronize();
		end = clock();
		t_str_radix8 = static_cast<double>(end-start)/CLOCKS_PER_SEC;
		std::cout << "Str<8> radix. Elapsed: " << t_str_radix8 << " sec. ";
		std::cout << "Faster than Str1: " << t_str1/t_str_radix8 << " X. ";
		h_i_result = d_indecies;
		std::cout << "Indexes " << (thrust::equal(h_i1.begin(), h_i1.end(), h_i_result.begin())?"equal":"differ");
		thrust::host_vector<T_str > h_v_result = d_v;
		std::cout << ",data " << (thrust::equal(h_v_result.data(), h_v_result.data() + ROWS_COUNT, (T_str *)h_v1.data())?"equal":"differ") << std::endl;
	}
		
	{
		typedef typename Str3<string_length> T_str;
		/// copy host to device
		thrust::device_vector<T_str > d_v(ROWS_COUNT);
		thrust::copy((T_str *)h_v.data(), (T_str *)h_v.data() + ROWS_COUNT, d_v.data());

		/// generate sequence indecies
		thrust::sequence(d_indecies.begin(), d_indecies.end());
		cudaThreadSynchronize();

		/// Test time for stable sort by key
		start = clock();
		thrust::stable_sort_by_key(d_v.begin(), d_v.end(), d_indecies.begin());
		cudaThreadSynchronize();
		end = clock();
		t_str3 = static_cast<double>(end-start)/CLOCKS_PER_SEC;
		std::cout << "Str3<" << string_length << "> Elapsed: " << t_str3 << " sec. ";
		std::cout << "Faster than Str1: " << t_str1/t_str3 << " X. ";
		h_i_result = d_indecies;
		std::cout << "Indexes " << (thrust::equal(h_i1.begin(), h_i1.end(), h_i_result.begin())?"equal":"differ");
		thrust::host_vector<T_str > h_v_result = d_v;
		std::cout << ",data " << (thrust::equal(h_v_result.data(), h_v_result.data() + ROWS_COUNT, (T_str *)h_v1.data())?"equal":"differ") << std::endl;
	}

	{
		typedef typename Str4<string_length> T_str;
		/// copy host to device
		thrust::device_vector<T_str > d_v(ROWS_COUNT);
		thrust::copy((T_str *)h_v.data(), (T_str *)h_v.data() + ROWS_COUNT, d_v.data());

		/// generate sequence indecies
		thrust::sequence(d_indecies.begin(), d_indecies.end());
		cudaThreadSynchronize();

		/// Test time for stable sort by key
		start = clock();
		thrust::stable_sort_by_key(d_v.begin(), d_v.end(), d_indecies.begin());
		cudaThreadSynchronize();
		end = clock();
		t_str4 = static_cast<double>(end-start)/CLOCKS_PER_SEC;
		std::cout << "Str4<" << string_length << "> Elapsed: " << t_str4 << " sec. ";
		std::cout << "Faster than Str1: " << t_str1/t_str4 << " X. ";
		h_i_result = d_indecies;
		std::cout << "Indexes " << (thrust::equal(h_i1.begin(), h_i1.end(), h_i_result.begin())?"equal":"differ");
		thrust::host_vector<T_str > h_v_result = d_v;
		std::cout << ",data " << (thrust::equal(h_v_result.data(), h_v_result.data() + ROWS_COUNT, (T_str *)h_v1.data())?"equal":"differ") << std::endl;
	}

	std::cout << "End ----------------------------------------" << std::endl;
}
// --------------------------------------------------------------------------------

int main() {
	std::cout << std::endl << "A comparison of optimized sorting options static strings." << std::endl;
	std::cout << "ROWS_COUNT = " << ROWS_COUNT << std::endl;

	std::cout << std::endl << "With all unique strings, cardinality = " << ROWS_COUNT << std::endl;

	// with all unique strings, cardinality = ROWS_COUNT
	test_case<4>();
	test_case<8>();
	test_case<10>();
	test_case<40>();
	test_case<50>();
	test_case<100>();
	std::cout << "=======================================================" << std::endl;

	const size_t cardinality = 1000;
	std::cout << std::endl << "With only number of unique strings equal to cardinality = " << cardinality << std::endl;
	// with only number of unique strings equal to cardinality = 1000
	test_case<4>(cardinality);
	test_case<8>(cardinality);
	test_case<10>(cardinality);
	test_case<40>(cardinality);
	test_case<50>(cardinality);
	test_case<100>(cardinality);
	std::cout << "=======================================================" << std::endl;


	// Test swap little endian to big endian
	unsigned long long a[2] = { 1 + (2<<(8*1)) + (3<<(8*2)) + (4<<(8*3)) + 
		((unsigned long long)5<<(8*4)) + ((unsigned long long)6<<(8*5)) + ((unsigned long long)7<<(8*6)) + ((unsigned long long)8<<(8*7)) }; //  
	unsigned char *ptr = reinterpret_cast<unsigned char *>(a);
	unsigned int *ptr_int = reinterpret_cast<unsigned int *>(a);
	std::cout << "*ptr_int = " << *ptr_int << std::endl;
	for(size_t i = 0; i < 8; ++i) std::cout << (unsigned)ptr[i] << ",  ";
	std::cout << std::endl;

	a[1] = T_swap_le_be_64().operator()(a[0]);
	for(size_t i = 0; i < 8; ++i) std::cout << (unsigned)ptr[i + 8] << ",  ";
	std::cout << std::endl;


	int b;
	std::cin >> b;
	return 0;
}