/// ======================================== H File =================================================
/**
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *	  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
//---------------------------------------------------------------------------
#pragma once
#ifndef STRINGS_TYPE_H
#define STRINGS_TYPE_H
//---------------------------------------------------------------------------
#pragma inline_recursion(on)
#pragma inline_depth(16)
#pragma GCC optimize ("unroll-loops")


/// Static string type 1
template<unsigned int N, typename T = unsigned char>
struct Str {
	T data[N];
	enum { size = N };

	__host__ __device__
	bool operator<(const Str& other) const
	{
		for(unsigned int i = 0; i < N ; i++) {
			if(data[i] > other.data[i]) {
				return 0;
			} else if(data[i] < other.data[i]) {
				return 1;
			}
		}
		return 0;
	}

	__host__ __device__
	bool operator>(const Str& other) const
	{
		for(unsigned int i = 0; i < N ; i++) {
			if(data[i] > other.data[i]) {
				return 1;
			} else if(data[i] < other.data[i]) {
				return 0;
			}
		}
		return 0;
	}


	__host__ __device__
	bool operator!=(const Str& other) const
	{
		for(unsigned int i = 0; i < N ; i++) {
			if(data[i] != other.data[i]) {
				return 1;
			}
		}
		return 0;
	}

	/// Additional comparisons
	__host__ __device__ bool operator>=(const Str& str) const { return !(*this < str); }
	__host__ __device__ bool operator<=(const Str& str) const { return !(*this > str); }
	__host__ __device__ bool operator==(const Str& str) const { return !(*this != str); }
};
// ---------------------------------------------------------------------------




/// Static string type 2
template<unsigned int N, typename T = unsigned char>
struct Str2 {
	T data[N];
	enum { size = N };

	__host__ __device__
	inline short int less(T const*const data, T const*const other_data) const {
		if(data[0] > other_data[0]) return 0;
		else if(data[0] < other_data[0]) return 1;
		if(data[1] > other_data[1]) return 0;
		else if(data[1] < other_data[1]) return 1;
		if(data[2] > other_data[2]) return 0;
		else if(data[2] < other_data[2]) return 1;
		if(data[3] > other_data[3]) return 0;
		else if(data[3] < other_data[3]) return 1;
		return 2;
	}

	__host__ __device__
	bool operator<(const Str2& other) const
	{
		
		/*
		/// Unrolls loops is especially important for CUDA-pipelines
		#pragma unroll N
		for(unsigned int i = 0; i < N ; i++) {
			if(data[i] > other.data[i]) {
				return 0;
			} else if(data[i] < other.data[i]) {
				return 1;
			}
		}*/

		T const* src_data = data;
		T const* other_data = other.data;
		for(T const* src_data = data; src_data < data+N; src_data+=4, other_data+=4) {
			if(src_data[0] > other_data[0]) return 0;
			else if(src_data[0] < other_data[0]) return 1;
			if(src_data[1] > other_data[1]) return 0;
			else if(src_data[1] < other_data[1]) return 1;
			if(src_data[2] > other_data[2]) return 0;
			else if(src_data[2] < other_data[2]) return 1;
			if(src_data[3] > other_data[3]) return 0;
			else if(src_data[3] < other_data[3]) return 1;
		}		
		return false;
	}

	__host__ __device__
	bool operator>(const Str2& other) const
	{
		#pragma unroll
		for(unsigned int i = 0; i < N ; i++) {
			if(data[i] > other.data[i]) {
				return 1;
			} else if(data[i] < other.data[i]) {
				return 0;
			}
		}
		return 0;
	}


	__host__ __device__
	bool operator!=(const Str2& other) const
	{
		#pragma unroll
		for(unsigned int i = 0; i < N ; i++) {
			if(data[i] != other.data[i]) {
				return 1;
			}
		}
		return 0;
	}

	/// Additional comparisons
	__host__ __device__ bool operator>=(const Str2& str) const { return !(*this < str); }
	__host__ __device__ bool operator<=(const Str2& str) const { return !(*this > str); }
	__host__ __device__ bool operator==(const Str2& str) const { return !(*this != str); }
};
// ---------------------------------------------------------------------------



/// Static string type 3
template<unsigned int N, typename T = unsigned char>
struct Str3 {
	T data[N];
	enum { size = N };
	enum { bytes2_count = size / sizeof(unsigned short int) };
	static_assert(sizeof(unsigned short int) == 2, "You can't use this optimized class, because it can't compare data by 2 bytes!");
	
	__host__ __device__
	inline unsigned short int swap_le_be_16(unsigned short int const& val) const {
		return (val<<8) |		// move byte 0 to byte 1
				(val>>8);		// move byte 1 to byte 0
	}

	__host__ __device__
	bool operator<(const Str3& other) const
	{
		unsigned short int const* data2_first = reinterpret_cast<unsigned short int const*>(data);
		unsigned short int const* data2_second = reinterpret_cast<unsigned short int const *>(other.data);
		
		#pragma unroll
		for(int i = 0; i < bytes2_count; i++) {
			if(swap_le_be_16(data2_first[i]) > swap_le_be_16(data2_second[i])) return false;
			else if(swap_le_be_16(data2_first[i]) < swap_le_be_16(data2_second[i])) return true;
		}
		
		#pragma unroll
		for(int i = bytes2_count*2; i < size; i++) {
			if(data[i] > other.data[i]) return false;
			else if(data[i] < other.data[i]) return true;
		}
		return false;
	}

	__host__ __device__
	bool operator>(const Str3& other) const
	{
		unsigned short int const* data2_first = reinterpret_cast<unsigned short int const*>(data);
		unsigned short int const* data2_second = reinterpret_cast<unsigned short int const *>(other.data);
		
		#pragma unroll
		for(int i = 0; i < bytes2_count; i++) {
			if(swap_le_be_16(data2_first[i]) > swap_le_be_16(data2_second[i])) return true;
			else if(swap_le_be_16(data2_first[i]) < swap_le_be_16(data2_second[i])) return false;
		}
		
		#pragma unroll
		for(int i = bytes2_count*2; i < size; i++) {
			if(data[i] > other.data[i]) return true;
			else if(data[i] < other.data[i]) return false;
		}
		return false;
	}


	__host__ __device__
	bool operator!=(const Str3& other) const
	{
		unsigned short int const* data2_first = reinterpret_cast<unsigned short int const*>(data);
		unsigned short int const* data2_second = reinterpret_cast<unsigned short int const*>(other.data);
		#pragma unroll
		for(unsigned int i = 0; i < bytes2_count; i++) {
			if(swap_le_be_16(data2_first[i]) != swap_le_be_16(data2_second[i])) return true;
		}

		#pragma unroll
		for(unsigned int i = bytes2_count*2; i < size; i++) {
			if(data[i] != other.data[i]) return true;
		}
		return false;
	}

	/// Additional comparisons
	__host__ __device__ bool operator>=(const Str3& str) const { return !(*this < str); }
	__host__ __device__ bool operator<=(const Str3& str) const { return !(*this > str); }
	__host__ __device__ bool operator==(const Str3& str) const { return !(*this != str); }
};
// ---------------------------------------------------------------------------


/// Unrolling the templated functor (reusable)
template<unsigned int unroll_count, unsigned int N = unroll_count >
struct T_unroll_compare_less {
	T_unroll_compare_less<unroll_count-1, N> next_unroll;		/// Next step of unrolling
	enum { index = N - unroll_count };

	__host__ __device__
	inline unsigned short int swap_le_be_16(unsigned short int const& val) const {
		return (val<<8) |		// move byte 0 to byte 1
				(val>>8);		// move byte 1 to byte 0
	}

	template<typename T1, typename T2>
	__host__ __device__
	__forceinline bool operator()(T1 const& data2_first, T2 const& data2_second, bool const& odd) const {
		if(swap_le_be_16(data2_first[index]) > swap_le_be_16(data2_second[index])) return false;
		else if(swap_le_be_16(data2_first[index]) < swap_le_be_16(data2_second[index])) return true;
		return next_unroll(data2_first, data2_second, odd);
	}
};
/// End of unroll (partial specialization)
template<unsigned int N>
struct T_unroll_compare_less<0, N> { 
	template<typename T1, typename T2>
	__host__ __device__ __forceinline bool operator()(T1 const& data2_first, T2 const& data2_second, bool const& odd) const { 
		if(odd) {
			unsigned char const* data_first = reinterpret_cast<unsigned char const*>(data2_first);
			unsigned char const* data_second = reinterpret_cast<unsigned char const*>(data2_second);
			if(data_first[N*2] > data_second[N*2]) return false;
			else if(data_first[N*2] < data_second[N*2]) return true;
		}
		return false; 
	}
};
// -----------------------------------------------------------------------


/// Static string type 4
template<unsigned int N, typename T = unsigned char>
struct Str4 {
	T data[N];
	enum { size = N };
	enum { bytes2_count = size / sizeof(unsigned short int), 
		bytes4_count = size / sizeof(unsigned int),
		bytes8_count = size / sizeof(unsigned long long) };

	__host__ __device__
	inline unsigned short int swap_le_be_16(unsigned short int const& val) const {
		return (val<<8) |		// move byte 0 to byte 1
				(val>>8);		// move byte 1 to byte 0
	}

	__host__ __device__
	inline unsigned int swap_le_be_32(unsigned int const& val) const {
		return ((val>>24)) |			// move byte 3 to byte 0
				((val<<8)&0xff0000) |	// move byte 1 to byte 2
				((val>>8)&0xff00) |		// move byte 2 to byte 1
				((val<<24));			// byte 0 to byte 3
	}

	__host__ __device__
	bool operator<(const Str4& other) const
	{		
		
		if(size % 4 == 0) {
			/// Speedup in 1.5 - 3.5 times (compare aligned data by 4 bytes)
			static_assert(sizeof(unsigned int) == 4, "You can't use this optimized class, because it can't compare data by 4 bytes!");
			unsigned int const* data4_first = reinterpret_cast<unsigned int const*>(data);
			unsigned int const* data4_second = reinterpret_cast<unsigned int const *>(other.data);
			#pragma unroll
			for(unsigned int i = 0; i < bytes4_count; i++) {
				if(swap_le_be_32(data4_first[i]) > swap_le_be_32(data4_second[i])) return false;
				else if(swap_le_be_32(data4_first[i]) < swap_le_be_32(data4_second[i])) return true;
			}
		} else 
		{
			
			/// Speedup in 1.5 - 2 times (compare unaligned data by 2 bytes)
			unsigned short int const* data2_first = reinterpret_cast<unsigned short int const*>(data);
			unsigned short int const* data2_second = reinterpret_cast<unsigned short int const *>(other.data);
		
				/*
			if(size > 16) {
				if(swap_le_be_16(data2_first[0]) > swap_le_be_16(data2_second[0])) return false;
				else if(swap_le_be_16(data2_first[0]) < swap_le_be_16(data2_second[0])) return true;
				if(swap_le_be_16(data2_first[1]) > swap_le_be_16(data2_second[1])) return false;
				else if(swap_le_be_16(data2_first[1]) < swap_le_be_16(data2_second[1])) return true;
				if(swap_le_be_16(data2_first[2]) > swap_le_be_16(data2_second[2])) return false;
				else if(swap_le_be_16(data2_first[2]) < swap_le_be_16(data2_second[2])) return true;
				if(swap_le_be_16(data2_first[3]) > swap_le_be_16(data2_second[3])) return false;
				else if(swap_le_be_16(data2_first[3]) < swap_le_be_16(data2_second[3])) return true;

				if(swap_le_be_16(data2_first[4]) > swap_le_be_16(data2_second[4])) return false;
				else if(swap_le_be_16(data2_first[4]) < swap_le_be_16(data2_second[4])) return true;
				if(swap_le_be_16(data2_first[5]) > swap_le_be_16(data2_second[5])) return false;
				else if(swap_le_be_16(data2_first[5]) < swap_le_be_16(data2_second[5])) return true;
				if(swap_le_be_16(data2_first[6]) > swap_le_be_16(data2_second[6])) return false;
				else if(swap_le_be_16(data2_first[6]) < swap_le_be_16(data2_second[6])) return true;
				if(swap_le_be_16(data2_first[7]) > swap_le_be_16(data2_second[7])) return false;
				else if(swap_le_be_16(data2_first[7]) < swap_le_be_16(data2_second[7])) return true;

			} else 
			if(size > 8) {
				if(swap_le_be_16(data2_first[0]) > swap_le_be_16(data2_second[0])) return false;
				else if(swap_le_be_16(data2_first[0]) < swap_le_be_16(data2_second[0])) return true;
				if(swap_le_be_16(data2_first[1]) > swap_le_be_16(data2_second[1])) return false;
				else if(swap_le_be_16(data2_first[1]) < swap_le_be_16(data2_second[1])) return true;
				if(swap_le_be_16(data2_first[2]) > swap_le_be_16(data2_second[2])) return false;
				else if(swap_le_be_16(data2_first[2]) < swap_le_be_16(data2_second[2])) return true;
				if(swap_le_be_16(data2_first[3]) > swap_le_be_16(data2_second[3])) return false;
				else if(swap_le_be_16(data2_first[3]) < swap_le_be_16(data2_second[3])) return true;
			}
			

			#pragma unroll
			for(unsigned int i = ((size>16)?8:((size>8)?4:0)); i < bytes2_count; i++) {
				if(swap_le_be_16(data2_first[i]) > swap_le_be_16(data2_second[i])) return false;
				else if(swap_le_be_16(data2_first[i]) < swap_le_be_16(data2_second[i])) return true;
			}*/

			return T_unroll_compare_less<bytes2_count>().operator()(data2_first, data2_second, size%2);

			#pragma unroll
			for(unsigned int i = bytes2_count*2; i < size; i++) {
				if(data[i] > other.data[i]) return false;
				else if(data[i] < other.data[i]) return true;
			}
		}
		return false;
	}

	__host__ __device__
	bool operator>(const Str4& other) const
	{
		if(size % 4 == 0) {
			/// Speedup in 1.5 - 3.5 times (compare aligned data by 4 bytes)
			unsigned int const* data4_first = reinterpret_cast<unsigned int const*>(data);
			unsigned int const* data4_second = reinterpret_cast<unsigned int const *>(other.data);
			#pragma unroll
			for(unsigned int i = 0; i < size/4; i++) {
				if(swap_le_be_32(data4_first[i]) > swap_le_be_32(data4_second[i])) return true;
				else if(swap_le_be_32(data4_first[i]) < swap_le_be_32(data4_second[i])) return false;
			}
		} else {
			/// Speedup in 1.5 - 2 times (compare unaligned data by 2 bytes)
			unsigned short int const* data2_first = reinterpret_cast<unsigned short int const*>(data);
			unsigned short int const* data2_second = reinterpret_cast<unsigned short int const *>(other.data);
		
			#pragma unroll
			for(unsigned int i = 0; i < bytes2_count; i++) {
				if(swap_le_be_16(data2_first[i]) > swap_le_be_16(data2_second[i])) return true;
				else if(swap_le_be_16(data2_first[i]) < swap_le_be_16(data2_second[i])) return false;
			}
		
			#pragma unroll
			for(unsigned int i = bytes2_count*2; i < size; i++) {
				if(data[i] > other.data[i]) return true;
				else if(data[i] < other.data[i]) return false;
			}
		}
		return false;
	}


	__host__ __device__
	bool operator!=(const Str4& other) const
	{
		if(size % 4 == 0) {
			unsigned int const* data4_first = reinterpret_cast<unsigned int const*>(data);
			unsigned int const* data4_second = reinterpret_cast<unsigned int const *>(other.data);
			#pragma unroll
			for(unsigned int i = 0; i < size/4; i++) {
				if(swap_le_be_32(data4_first[i]) != swap_le_be_32(data4_second[i])) return true;
			}
		} else {
			unsigned short int const* data2_first = reinterpret_cast<unsigned short int const*>(data);
			unsigned short int const* data2_second = reinterpret_cast<unsigned short int const*>(other.data);
			#pragma unroll
			for(unsigned int i = 0; i < bytes2_count; i++) {
				if(swap_le_be_16(data2_first[i]) != swap_le_be_16(data2_second[i])) return true;
			}

			#pragma unroll
			for(unsigned int i = bytes2_count*2; i < size; i++) {
				if(data[i] != other.data[i]) return true;
			}
		}
		return false;
	}

	/// Additional comparisons
	__host__ __device__ bool operator>=(const Str4& str) const { return !(*this < str); }
	__host__ __device__ bool operator<=(const Str4& str) const { return !(*this > str); }
	__host__ __device__ bool operator==(const Str4& str) const { return !(*this != str); }
};
//

//---------------------------------------------------------------------------
#endif	/// STRINGS_TYPE_H