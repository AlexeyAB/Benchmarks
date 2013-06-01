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
//---------------------------------------------------------------------------


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
	enum { bytes2_count = size / sizeof(unsigned short int) };
	static_assert(sizeof(unsigned short int) == 2, "You can't use this optimized class, because it can't compare data by 2 bytes!");
	
	__host__ __device__
	inline unsigned short int swap_le_be_16(unsigned short int const& val) const {
		return (val<<8) |		// move byte 0 to byte 1
				(val>>8);		// move byte 1 to byte 0
	}

	__host__ __device__
	bool operator<(const Str2& other) const
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
	bool operator>(const Str2& other) const
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
	bool operator!=(const Str2& other) const
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
	enum { bytes2_count = size / sizeof(unsigned short int), 
		bytes4_count = size / sizeof(unsigned int) };

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


	/// Uniform comparison
	__host__ __device__
	unsigned char comparison(const Str3& other) const {
			if(size % 4 == 0) {
			/// Speedup in 1.5 - 3.5 times (compare aligned data by 4 bytes)
			static_assert(sizeof(unsigned int) == 4, "You can't use this optimized class, because it can't compare data by 4 bytes!");
			unsigned int const* data4_first = reinterpret_cast<unsigned int const*>(data);
			unsigned int const* data4_second = reinterpret_cast<unsigned int const *>(other.data);
			#pragma unroll
			for(unsigned int i = 0; i < bytes4_count; i++) {
				if(swap_le_be_32(data4_first[i]) > swap_le_be_32(data4_second[i])) return 0;
				else if(swap_le_be_32(data4_first[i]) < swap_le_be_32(data4_second[i])) return 1;
			}
		} else {			
			/// Speedup in 1.5 - 2 times (compare unaligned data by 2 bytes)
			unsigned short int const* data2_first = reinterpret_cast<unsigned short int const*>(data);
			unsigned short int const* data2_second = reinterpret_cast<unsigned short int const *>(other.data);
				
			#pragma unroll
			for(unsigned int i = 0; i < bytes2_count; i++) {
				if(swap_le_be_16(data2_first[i]) > swap_le_be_16(data2_second[i])) return 0;
				else if(swap_le_be_16(data2_first[i]) < swap_le_be_16(data2_second[i])) return 1;
			}

			#pragma unroll
			for(unsigned int i = bytes2_count*2; i < size; i++) {
				if(data[i] > other.data[i]) return 0;
				else if(data[i] < other.data[i]) return 1;
			}			
		}
		return 2;
	}
	

	__host__ __device__
	bool operator<(const Str3& other) const
	{		
		unsigned char ret = comparison(other);
		if(ret != 2) return ret;
		else return false;
	}

	__host__ __device__
	bool operator>(const Str3& other) const
	{
		unsigned char ret = comparison(other);
		if(ret != 2) return !ret;
		else return false;
	}


	__host__ __device__
	bool operator!=(const Str3& other) const
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
	__host__ __device__ bool operator>=(const Str3& str) const { return !(*this < str); }
	__host__ __device__ bool operator<=(const Str3& str) const { return !(*this > str); }
	__host__ __device__ bool operator==(const Str3& str) const { return !(*this != str); }
};
// ---------------------------------------------------------------------------



/// Static string type 4
template<unsigned int N, typename T = unsigned char>
struct Str4 {
	T data[N];
	enum { size = N };
	enum { bytes2_count = size / sizeof(unsigned short int), 
		bytes4_count = size / sizeof(unsigned int) };

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


	/// Uniform comparison
	__host__ __device__
	unsigned char comparison(const Str4& other) const {
			if(size % 4 == 0) {
			/// Speedup in 1.5 - 3.5 times (compare aligned data by 4 bytes)
			static_assert(sizeof(unsigned int) == 4, "You can't use this optimized class, because it can't compare data by 4 bytes!");
			unsigned int const* data4_first = reinterpret_cast<unsigned int const*>(data);
			unsigned int const* data4_second = reinterpret_cast<unsigned int const *>(other.data);
			#pragma unroll
			for(unsigned int i = 0; i < bytes4_count; i++) {
				if(swap_le_be_32(data4_first[i]) > swap_le_be_32(data4_second[i])) return 0;
				else if(swap_le_be_32(data4_first[i]) < swap_le_be_32(data4_second[i])) return 1;
			}
		} else {			
			/// Speedup in 2 - 2.5 times (compare unaligned data by 2 bytes)
			unsigned short int const* data2_first = reinterpret_cast<unsigned short int const*>(data);
			unsigned short int const* data2_second = reinterpret_cast<unsigned short int const *>(other.data);

			if(size > 16) {
				if(swap_le_be_16(data2_first[0]) > swap_le_be_16(data2_second[0])) return 0;
				else if(swap_le_be_16(data2_first[0]) < swap_le_be_16(data2_second[0])) return 1;
				if(swap_le_be_16(data2_first[1]) > swap_le_be_16(data2_second[1])) return 0;
				else if(swap_le_be_16(data2_first[1]) < swap_le_be_16(data2_second[1])) return 1;
				if(swap_le_be_16(data2_first[2]) > swap_le_be_16(data2_second[2])) return 0;
				else if(swap_le_be_16(data2_first[2]) < swap_le_be_16(data2_second[2])) return 1;
				if(swap_le_be_16(data2_first[3]) > swap_le_be_16(data2_second[3])) return 0;
				else if(swap_le_be_16(data2_first[3]) < swap_le_be_16(data2_second[3])) return 1;

				if(swap_le_be_16(data2_first[4]) > swap_le_be_16(data2_second[4])) return 0;
				else if(swap_le_be_16(data2_first[4]) < swap_le_be_16(data2_second[4])) return 1;
				if(swap_le_be_16(data2_first[5]) > swap_le_be_16(data2_second[5])) return 0;
				else if(swap_le_be_16(data2_first[5]) < swap_le_be_16(data2_second[5])) return 1;
				if(swap_le_be_16(data2_first[6]) > swap_le_be_16(data2_second[6])) return 0;
				else if(swap_le_be_16(data2_first[6]) < swap_le_be_16(data2_second[6])) return 1;
				if(swap_le_be_16(data2_first[7]) > swap_le_be_16(data2_second[7])) return 0;
				else if(swap_le_be_16(data2_first[7]) < swap_le_be_16(data2_second[7])) return 1;
			} else 
			if(size > 8) {
				if(swap_le_be_16(data2_first[0]) > swap_le_be_16(data2_second[0])) return 0;
				else if(swap_le_be_16(data2_first[0]) < swap_le_be_16(data2_second[0])) return 1;
				if(swap_le_be_16(data2_first[1]) > swap_le_be_16(data2_second[1])) return 0;
				else if(swap_le_be_16(data2_first[1]) < swap_le_be_16(data2_second[1])) return 1;
				if(swap_le_be_16(data2_first[2]) > swap_le_be_16(data2_second[2])) return 0;
				else if(swap_le_be_16(data2_first[2]) < swap_le_be_16(data2_second[2])) return 1;
				if(swap_le_be_16(data2_first[3]) > swap_le_be_16(data2_second[3])) return 0;
				else if(swap_le_be_16(data2_first[3]) < swap_le_be_16(data2_second[3])) return 1;
			}

			#pragma unroll
			for(unsigned int i = ((size>16)?8:((size>8)?4:0)); i < bytes2_count; i++) {
				if(swap_le_be_16(data2_first[i]) > swap_le_be_16(data2_second[i])) return 0;
				else if(swap_le_be_16(data2_first[i]) < swap_le_be_16(data2_second[i])) return 1;
			}

			#pragma unroll
			for(unsigned int i = bytes2_count*2; i < size; i++) {
				if(data[i] > other.data[i]) return 0;
				else if(data[i] < other.data[i]) return 1;
			}			
		}
		return 2;
	}
	

	__host__ __device__
	bool operator<(const Str4& other) const
	{		
		unsigned char ret = comparison(other);
		if(ret != 2) return ret;
		else return false;
	}

	__host__ __device__
	bool operator>(const Str4& other) const
	{
		unsigned char ret = comparison(other);
		if(ret != 2) return !ret;
		else return false;
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
//---------------------------------------------------------------------------



/// Static string type 4
template<unsigned int N, typename T = unsigned char>
struct Str5 {
	T data[N];
	enum { size = N };
	enum { bytes2_count = size / sizeof(unsigned short int), 
		bytes4_count = size / sizeof(unsigned int) };

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
	//---------------------------------------------------------------------------

	/// Unrolling the templated functor (reusable)
	template<unsigned int unroll_count, unsigned int N = unroll_count >
	struct T_unroll_compare_less {
		T_unroll_compare_less<unroll_count-1, N> next_unroll;		/// Next step of unrolling
		enum { index = N - unroll_count };

		__host__ __device__
		inline unsigned short int swap_le_be_16(unsigned short int const& val) const { 
			return (val<<8) | (val>>8); 
		}

		template<typename T1, typename T2>
		__host__ __device__
		inline unsigned char operator()(T1 const& data2_first, T2 const& data2_second, bool const& odd) const {
			if(swap_le_be_16(data2_first[index]) > swap_le_be_16(data2_second[index])) return 0;
			else if(swap_le_be_16(data2_first[index]) < swap_le_be_16(data2_second[index])) return 1;
			return next_unroll(data2_first, data2_second, odd);
		}
	};
	/// End of unroll (partial specialization)
	template<unsigned int N>
	struct T_unroll_compare_less<0, N> { 
		template<typename T1, typename T2>
		__host__ __device__ inline unsigned char operator()(T1 const& data2_first, T2 const& data2_second, bool const& odd) const { 
			if(odd) {
				unsigned char const* data_first = reinterpret_cast<unsigned char const*>(data2_first);
				unsigned char const* data_second = reinterpret_cast<unsigned char const*>(data2_second);
				if(data_first[N*2] > data_second[N*2]) return 0;
				else if(data_first[N*2] < data_second[N*2]) return 1;
			}
			return 0; 
		}
	};
	// -----------------------------------------------------------------------


	/// Uniform comparison
	__host__ __device__
	unsigned char comparison(const Str5& other) const {
			if(size % 4 == 0) {
			/// Speedup in 1.5 - 3.5 times (compare aligned data by 4 bytes)
			static_assert(sizeof(unsigned int) == 4, "You can't use this optimized class, because it can't compare data by 4 bytes!");
			unsigned int const* data4_first = reinterpret_cast<unsigned int const*>(data);
			unsigned int const* data4_second = reinterpret_cast<unsigned int const *>(other.data);
			#pragma unroll
			for(unsigned int i = 0; i < bytes4_count; i++) {
				if(swap_le_be_32(data4_first[i]) > swap_le_be_32(data4_second[i])) return 0;
				else if(swap_le_be_32(data4_first[i]) < swap_le_be_32(data4_second[i])) return 1;
			}
		} else {			
			/// Speedup in 2 - 2.5 times (compare unaligned data by 2 bytes)
			unsigned short int const* data2_first = reinterpret_cast<unsigned short int const*>(data);
			unsigned short int const* data2_second = reinterpret_cast<unsigned short int const *>(other.data);

			return T_unroll_compare_less<bytes2_count>().operator()(data2_first, data2_second, size%2);		
		}
		return 2;
	}
	

	__host__ __device__
	bool operator<(const Str5& other) const
	{		
		unsigned char ret = comparison(other);
		if(ret != 2) return ret;
		else return false;
	}

	__host__ __device__
	bool operator>(const Str5& other) const
	{
		unsigned char ret = comparison(other);
		if(ret != 2) return !ret;
		else return false;
	}


	__host__ __device__
	bool operator!=(const Str5& other) const
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
	__host__ __device__ bool operator>=(const Str5& str) const { return !(*this < str); }
	__host__ __device__ bool operator<=(const Str5& str) const { return !(*this > str); }
	__host__ __device__ bool operator==(const Str5& str) const { return !(*this != str); }
};
//---------------------------------------------------------------------------



//---------------------------------------------------------------------------
#endif	/// STRINGS_TYPE_H