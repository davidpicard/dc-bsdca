/**
    This file is part of JkernelMachines.

    JkernelMachines is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    JkernelMachines is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with JkernelMachines.  If not, see <http://www.gnu.org/licenses/>.

    Copyright David Picard - 2014

 */
package budget;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import net.jkernelmachines.density.DensityFunction;
import net.jkernelmachines.util.algebra.VectorOperations;

/**
 * Very basic KMeans algorithm with a shifting codeword procedure to ensure no
 * empty cluster and balanced distortion
 * 
 * @author picard
 * 
 */
public class OnlineDoubleKMeans implements DensityFunction<double[]> {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -2999550073143522736L;

	class Cluster {
		double[] mu;
		
		public Cluster(double[] x) {
			mu = Arrays.copyOf(x, x.length);
		}
		
		public void update(double[] x, double alpha) {
			VectorOperations.muli(mu, mu, 1-alpha);
			VectorOperations.addi(mu, mu, alpha, x);
		}
	}

	int K;
	List<Cluster> clusters;
	double alpha = 0.1;

	double shiftRatio = 20;

//	DebugPrinter debug = new DebugPrinter();

	/**
	 * Constructor with number of clusters
	 * 
	 * @param k
	 *            number of clusters
	 */
	public OnlineDoubleKMeans(int k) {
		K = k;
		clusters = new ArrayList<>(K);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.jkernelmachines.density.DensityFunction#train(java.lang.Object)
	 */
	@Override
	public void train(double[] e) {
		if(clusters.size() < K) {
			clusters.add(new Cluster(e));
		}
		else {
			clusters.get((int)valueOf(e)).update(e, alpha);
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.jkernelmachines.density.DensityFunction#train(java.util.List)
	 */
	@Override
	public void train(List<double[]> train) {
		for(double[] e : train) {
			train(e);
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.jkernelmachines.density.DensityFunction#valueOf(java.lang.Object)
	 */
	@Override
	public double valueOf(double[] e) {
		double dmin = Double.POSITIVE_INFINITY;
		int index = -1;
		for (int g = 0; g < clusters.size(); g++) {
			double d = VectorOperations.d2p2(e, clusters.get(g).mu);
			if (d < dmin) {
				dmin = d;
				index = g;
			}
		}
		return index;
	}

	/**
	 * Return an array containing the squared distances to each clusters
	 * 
	 * @param e
	 *            the sample to evaluate
	 * @return the array of distances
	 */
	public double[] distanceToMean(double[] e) {
		double[] d = new double[K];
		for (int g = 0; g < clusters.size(); g++) {
			d[g] = VectorOperations.d2p2(e, clusters.get(g).mu);
		}
		return d;
	}

	public double getAlpha() {
		return alpha;
	}

	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}
}
