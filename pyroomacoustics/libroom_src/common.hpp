/*
 * Type and constant definitions
 * Copyright (C) 2019  Robin Scheibler
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * You should have received a copy of the MIT License along with this program. If
 * not, see <https://opensource.org/licenses/MIT>.
 */

/* This file contains type and constants definitions */
#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <iostream>
#include <vector>
#include <list>
#include <cstdint>
#include <Eigen/Dense>

extern float libroom_eps;  // epsilon is the precision for floating point computations. It is defined in libroom.cpp

template<size_t D>
using Vectorf = Eigen::Matrix<float, D, 1>;
template<size_t D>
using Vectori = Eigen::Matrix<int, D, 1>;

using MatrixXf = Eigen::MatrixXf;
typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> MatrixXb;
typedef Eigen::Matrix<bool, Eigen::Dynamic, 1> VectorXb;

/* The 'entry' type is simply defined as an array of 2 floats.
 * It represents an entry that is logged by the microphone
 * during the ray_tracing execution.
 * The first one of those float will be the travel time of a ray reaching
 * the microphone. The second one will be the energy of this ray.*/

struct Hit
{
  float distance = 0.f;
  Eigen::ArrayXf transmitted;  // vector of transmitted energy over frequency bands

  Hit(int _nfreq)
  {
    transmitted.resize(_nfreq);
    transmitted.setOnes();
  };
  Hit(const float _d, const Eigen::ArrayXf &_t)
    : distance(_d), transmitted(_t) {}
};

typedef std::vector<std::list<Hit>> HitLog;

// NOTE: marked inline. As a non-inline free function in a header this is an
// ODR violation if common.hpp is included from more than one translation
// unit; it happens to link today only because it is currently included from
// one place.
inline size_t get_new_size(size_t val, size_t cur_size)
{
  size_t new_size = cur_size;
  while (val >= new_size)
    new_size *= 2;
  return new_size;
}

/*
 * ============================================================
 *  PRECISION FIX  --  why the accumulator is double, not float
 * ============================================================
 *
 * Ray tracing deposits energy into these bins one ray at a time, and the
 * energy carried by a single ray is ENERGY_0 / n_rays. The number of
 * deposits landing in a given bin therefore grows in proportion to the
 * ray count, while each deposit shrinks in proportion to it.
 *
 * Sequential floating point accumulation fails once the running total
 * grows large relative to an individual deposit: when
 *
 *     running_sum * epsilon  >  deposit
 *
 * the addition rounds to no change at all and the deposit is silently
 * discarded. For a stream of roughly equal deposits, that happens after
 * about 1/epsilon of them -- INDEPENDENT of how big the deposits are.
 * Making every deposit smaller does not help, because the running sum
 * shrinks by the same factor.
 *
 *     float32:  1/eps ~= 8.4e6   deposits before the bin stops growing
 *     float64:  1/eps ~= 4.5e15  deposits
 *
 * Measured, summing n deposits of size 1/n (exact answer 1.0):
 *
 *     n = 1e6   float32 -> 1.009      (fine)
 *     n = 1e7   float32 -> 1.065      (fine)
 *     n = 1e8   float32 -> 0.250      77% of deposits lost
 *     n = 1e9   float32 -> 0.031      97% of deposits lost
 *
 * So a run at 1e8 rays silently loses most of the energy in any heavily
 * populated bin, while a run at 1e7 does not -- the failure appears as
 * the simulation getting WORSE as the ray count is raised.
 *
 * This hits the diffuse component hardest. Room<D>::scat_ray() logs a
 * contribution at EVERY wall hit of EVERY ray, whereas the specular path
 * in simul_ray() only logs on the rare occasions a ray passes within
 * mic_radius. Diffuse-dominated bins therefore reach the stagnation
 * count first and stop accumulating, while sparse specular deposits keep
 * landing normally -- the impulse response loses diffuse energy
 * preferentially, and loses more of it the more rays you throw at it.
 *
 * The accumulator's precision is what matters here, not the deposit's:
 * the failing operation is `running_sum + deposit == running_sum`. Each
 * individual deposit is perfectly representable in float32. Hence
 * `array` becomes ArrayXXd. The .cast<double>() calls on the incoming
 * values below are only there because Eigen requires matching operand
 * types -- they are a compilation requirement, not the mechanism, and
 * adding them without widening `array` would change nothing.
 *
 * `counts` is widened to 64-bit for the same class of reason: at 1e8
 * rays with tens of bounces each, a busy bin can approach the ~2.1e9
 * ceiling of int32, after which the count wraps negative and bin()
 * returns garbage.
 */

class Histogram2D
{
  size_t rows, cols;

  // Accumulator in DOUBLE precision -- see the note above. This is the
  // actual fix; everything else in this class follows from it.
  Eigen::ArrayXXd array;

  // 64-bit counts: int32 can wrap at high ray counts.
  typedef Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic> ArrayXXi64;
  ArrayXXi64 counts;

  public:
    Histogram2D() {}  // empty constructor
    Histogram2D(int _r, int _c) : rows(_r), cols(_c)
    {
      init(rows, cols);
    }

    void init(int rows, int cols)
    {
      array.resize(rows, cols);
      array.setZero();
      counts.resize(rows, cols);
      counts.setZero();
    }

    void reset()
    {
      array.setZero();
      counts.setZero();
    }

    void resize_rows(int new_rows)
    {
      auto old_rows = array.rows();
      // this will resize the array while preserving the content
      array.conservativeResize(new_rows, Eigen::NoChange);
      counts.conservativeResize(new_rows, Eigen::NoChange);
      // We need to initialize the new elements
      if (new_rows > old_rows)
      {
        array.bottomRows(new_rows - old_rows).setZero();
        counts.bottomRows(new_rows - old_rows).setZero();
      }
    }

    void resize_cols(int new_cols)
    {
      auto old_cols = array.cols();
      // this will resize the array while preserving the content
      array.conservativeResize(Eigen::NoChange, new_cols);
      counts.conservativeResize(Eigen::NoChange, new_cols);
      // We need to initialize the new elements
      if (new_cols > old_cols)
      {
        array.rightCols(new_cols - old_cols).setZero();
        counts.rightCols(new_cols - old_cols).setZero();
      }
    }

    void log(Eigen::Index row, Eigen::Index col, float val)
    {
      if (row >= array.rows())
        resize_rows(get_new_size(row, array.rows()));

      if (col >= array.cols())
        resize_cols(get_new_size(col, array.cols()));

      array.coeffRef(row, col) += double(val);
      counts.coeffRef(row, col)++;
    }

    void log_col(Eigen::Index col, const Eigen::ArrayXf &val)
    {
      if (col >= array.cols())
        resize_cols(get_new_size(col, array.cols()));

      array.col(col) += val.cast<double>();
      counts.col(col) += 1;
    }

    void log_row(Eigen::Index row, const Eigen::ArrayXf &val)
    {
      if (row >= array.rows())
        resize_rows(get_new_size(row, array.rows()));

      array.row(row) += val.cast<double>();
      counts.row(row) += 1;
    }

    float bin(Eigen::Index row, Eigen::Index col) const
    {
      if (counts.coeff(row, col) != 0)
        return float(array.coeff(row, col) / double(counts.coeff(row, col)));
      else
        return 0.f;
    }

    // Kept returning ArrayXXf so existing callers and the Python binding
    // are unaffected. The accumulation happened in double; only the
    // final result is narrowed, which is harmless -- a single conversion
    // of an already-correct total, not a running sum.
    Eigen::ArrayXXf get_hist() const
    {
      return array.cast<float>();
    }

    // Full-precision accessor, for anyone who wants the totals without
    // the narrowing above.
    const Eigen::ArrayXXd &get_hist_double() const
    {
      return array;
    }

    // NOTE: this now returns 64-bit counts. If a pybind11 binding
    // declared this as ArrayXXi it will need updating; the numpy array
    // handed to Python becomes int64 instead of int32.
    const ArrayXXi64 &get_counts() const
    {
      return counts;
    }
};

#endif // __COMMON_HPP__
