#include<stdlib.h>
#include "q1.h"
#include "q2.h"
#include "q3.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

bool is_prime(long long n) {
    if (n < 2) return false;
    for (long long i = 2; i <= sqrt(n); i++) {
        if (n % i == 0) return false;
    }
    return true;
}

void find_rprimes(long long seed, vector<long long> & rprimes) {
    for (auto r : {1, 3, 7, 9}) {
        auto cand = 10 * seed + r;
        if (is_prime(cand)) {
            // cout << cand << endl;
            rprimes.push_back(cand);
            find_rprimes(cand, rprimes);
        }
    }
}

vector<long long> find_rprimes_wrapper() {
    vector<long long> rprimes = {2, 3, 5, 7};
    for (long long seed : {2, 3, 5, 7}) {
        find_rprimes(seed, rprimes);
    }
    return rprimes;
}

int main() {
    auto primes = find_rprimes_wrapper();
    auto maxx = max_element(primes.begin(), primes.end());
    auto largest = primes[distance(primes.begin(), maxx)];
    cout << "Largest rprime is " << largest << "." << endl;
}
