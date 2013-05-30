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
	bool operator<(const Str2& other) const
	{
		/// Unrolls loops is especially important for CUDA-pipelines
		#pragma unroll
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


/// Static string type 4
template<unsigned int N, typename T = unsigned char>
struct Str4 {
	T data[N];
	enum { size = N };
	enum { bytes2_count = size / sizeof(unsigned short int) };

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
			for(unsigned int i = 0; i < size/4; i++) {
				if(swap_le_be_32(data4_first[i]) > swap_le_be_32(data4_second[i])) return false;
				else if(swap_le_be_32(data4_first[i]) < swap_le_be_32(data4_second[i])) return true;
			}
		} else {
			/// Speedup in 1.5 - 2 times (compare unaligned data by 2 bytes)
			unsigned short int const* data2_first = reinterpret_cast<unsigned short int const*>(data);
			unsigned short int const* data2_second = reinterpret_cast<unsigned short int const *>(other.data);
		
			#pragma unroll
			for(unsigned int i = 0; i < bytes2_count; i++) {
				if(swap_le_be_16(data2_first[i]) > swap_le_be_16(data2_second[i])) return false;
				else if(swap_le_be_16(data2_first[i]) < swap_le_be_16(data2_second[i])) return true;
			}
		
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