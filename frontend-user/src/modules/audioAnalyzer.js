import { Logger } from '../utils/logger.js';

const logger = new Logger('AudioAnalyzer');

/**
 * 音频分析器 - 负责音频的频谱分析、基频检测和倍频计算
 */
export class AudioAnalyzer {
  constructor(audioContext) {
    this.audioContext = audioContext;
  }

  /**
   * 分析音频数据
   * @param {Float32Array} audioData - 音频采样数据
   * @param {number} sampleRate - 采样率
   * @param {number} fftSize - FFT 大小
   * @returns {Object} 分析结果
   */
  async analyze(audioData, sampleRate, fftSize = 8192) {
    logger.info('开始频谱分析', { dataLength: audioData.length, sampleRate, fftSize });

    // 执行 FFT 分析
    const frequencyData = this.performFFT(audioData, fftSize);
    
    // 计算频率分辨率
    const frequencyResolution = sampleRate / fftSize;
    
    // 生成频率数组
    const frequencies = [];
    const magnitudes = [];
    const binCount = fftSize / 2;
    
    for (let i = 0; i < binCount; i++) {
      const freq = i * frequencyResolution;
      if (freq > 20 && freq < 20000) { // 人耳可听范围
        frequencies.push(freq);
        magnitudes.push(frequencyData[i]);
      }
    }

    // 检测基频
    const fundamentalFreq = this.detectFundamentalFrequency(audioData, sampleRate, frequencies, magnitudes);
    
    // 计算倍频 (最大13倍)
    const harmonics = this.calculateHarmonics(fundamentalFreq, 13);
    
    // 过滤只保留基频和倍频附近的数据
    const filteredData = this.filterHarmonics(frequencies, magnitudes, fundamentalFreq, harmonics);
    
    // 计算频率区域数据
    const frequencyBands = this.calculateFrequencyBands(fundamentalFreq, harmonics, filteredData);
    
    // 计算声强随时间变化的热力图数据
    const heatmapData = this.calculateHeatmapData(audioData, sampleRate, fftSize, fundamentalFreq, harmonics);

    // 找出频率范围
    const minFreq = fundamentalFreq * 0.8;
    const maxFreq = Math.min(fundamentalFreq * 13.5, 20000);

    return {
      fundamentalFreq,
      harmonics,
      frequencies: filteredData.frequencies,
      magnitudes: filteredData.magnitudes,
      frequencyBands,
      heatmapData,
      minFreq,
      maxFreq,
      rawFrequencies: frequencies,
      rawMagnitudes: magnitudes
    };
  }

  /**
   * 执行 FFT 变换
   */
  performFFT(audioData, fftSize) {
    // 使用 Web Audio API 的 AnalyserNode 进行 FFT
    // 这里我们手动实现简化版 FFT
    const paddedData = new Float32Array(fftSize);
    const copyLength = Math.min(audioData.length, fftSize);
    
    // 应用汉宁窗（窗长始终以 FFT 大小为准，零填充部分保持为 0）
    for (let i = 0; i < copyLength; i++) {
      const window = 0.5 * (1 - Math.cos(2 * Math.PI * i / (fftSize - 1)));
      paddedData[i] = audioData[i] * window;
    }

    // 执行 FFT
    const fftResult = this.fft(paddedData);
    
    // 计算幅度谱
    const magnitudes = new Float32Array(fftSize / 2);
    for (let i = 0; i < fftSize / 2; i++) {
      const real = fftResult.real[i];
      const imag = fftResult.imag[i];
      magnitudes[i] = Math.sqrt(real * real + imag * imag);
    }

    return magnitudes;
  }

  /**
   * FFT 实现 (Cooley-Tukey 算法)
   */
  fft(data) {
    const n = data.length;
    
    if (n <= 1) {
      return { real: [data[0] || 0], imag: [0] };
    }

    // 复制输入数据
    const real = new Float32Array(n);
    const imag = new Float32Array(n);
    
    for (let i = 0; i < n; i++) {
      real[i] = data[i];
      imag[i] = 0;
    }

    // 位反转重排（DIT Cooley-Tukey 要求输入按位反转排列）
    for (let i = 0, j = 0; i < n; i++) {
      if (j > i) {
        const tmp = real[i];
        real[i] = real[j];
        real[j] = tmp;
      }
      let bit = n >> 1;
      while (j & bit) {
        j ^= bit;
        bit >>= 1;
      }
      j ^= bit;
    }

    // 迭代 FFT
    for (let size = 2; size <= n; size *= 2) {
      const halfSize = size / 2;

      for (let i = 0; i < n; i += size) {
        for (let j = 0; j < halfSize; j++) {
          // 旋转因子 W_size^j = exp(-2πi·j/size)
          const angle = -2 * Math.PI * j / size;
          const cos = Math.cos(angle);
          const sin = Math.sin(angle);

          const idx1 = i + j;
          const idx2 = i + j + halfSize;

          const tReal = real[idx2] * cos - imag[idx2] * sin;
          const tImag = real[idx2] * sin + imag[idx2] * cos;

          real[idx2] = real[idx1] - tReal;
          imag[idx2] = imag[idx1] - tImag;
          real[idx1] = real[idx1] + tReal;
          imag[idx1] = imag[idx1] + tImag;
        }
      }
    }

    return { real, imag };
  }

  /**
   * 检测基频 - 使用归一化自相关法，并用频谱能量做倍频/半频校验
   *
   * 古琴这类拨弦乐器的信号由整倍数谐波构成，朴素自相关在基频周期的
   * 整数倍处都会出现相关峰，加上信号幅度随时间衰减带来的窗口偏差，
   * 很容易把 2 倍周期（即低八度的半频）误选为基频。
   * 这里改为：
   *   1. 去均值 + 逐滞后归一化的自相关 (NACF)，消除窗口长度偏差；
   *   2. 选择第一个超过阈值的局部峰（最短周期），而不是全局最大值；
   *   3. 再用频谱能量校验：候选频率处几乎没有能量而 2 倍频能量显著时，
   *      判定为半频误检并上移一个八度。
   */
  detectFundamentalFrequency(audioData, sampleRate, frequencies, magnitudes) {
    // 方法1: 归一化自相关法
    const autocorrFreq = this.autocorrelation(audioData, sampleRate);

    // 方法2: 峰值检测法
    const peakFreq = this.findDominantPeak(frequencies, magnitudes);

    // 优先使用自相关法的结果，因为它对古琴这类谐波丰富的乐器更稳定
    let fundamentalFreq = autocorrFreq;

    // 如果自相关法结果不合理，使用峰值检测
    if (!fundamentalFreq || fundamentalFreq < 50 || fundamentalFreq > 2000) {
      fundamentalFreq = peakFreq;
    }

    // 频谱校验：修正半频（低八度）误检
    fundamentalFreq = this.correctOctaveError(fundamentalFreq, frequencies, magnitudes);

    logger.info('基频检测结果', { autocorrFreq, peakFreq, final: fundamentalFreq });

    return fundamentalFreq;
  }

  /**
   * 归一化自相关法检测基频
   *
   * 对每个滞后量分别去均值并按两段能量归一化，得到取值 [-1, 1] 的
   * 相关系数，避免长滞后因参与求和样本少而虚高；随后选取第一个
   * 超过阈值的显著峰，从源头避免把基频周期的整数倍当成周期。
   */
  autocorrelation(audioData, sampleRate) {
    const minPeriod = Math.floor(sampleRate / 2000); // 最高频率 2000Hz
    const maxPeriodCandidate = Math.floor(sampleRate / 50);   // 最低频率 50Hz
    const dataLength = Math.min(audioData.length, sampleRate); // 最多分析1秒

    // 至少要容纳两个最短周期；滞后量不能超过数据长度的一半
    const maxPeriod = Math.min(maxPeriodCandidate, Math.floor(dataLength / 2));
    if (dataLength < 2 * minPeriod || maxPeriod <= minPeriod) {
      return 0;
    }

    // 前缀和，用于 O(1) 计算各滞后段的均值
    const prefixSum = new Float64Array(dataLength + 1);
    for (let i = 0; i < dataLength; i++) {
      prefixSum[i + 1] = prefixSum[i] + audioData[i];
    }

    const nacf = new Float64Array(maxPeriod);
    let globalMax = 0;

    for (let period = minPeriod; period < maxPeriod; period++) {
      const overlap = dataLength - period;

      // 去掉两段各自的直流分量后再做相关
      const mean1 = (prefixSum[overlap] - prefixSum[0]) / overlap;
      const mean2 = (prefixSum[dataLength] - prefixSum[period]) / overlap;
      let cov = 0;
      for (let i = 0; i < overlap; i++) {
        cov += (audioData[i] - mean1) * (audioData[i + period] - mean2);
      }

      let var1 = 0;
      let var2 = 0;
      for (let i = 0; i < overlap; i++) {
        const d1 = audioData[i] - mean1;
        const d2 = audioData[i + period] - mean2;
        var1 += d1 * d1;
        var2 += d2 * d2;
      }

      const denom = Math.sqrt(var1 * var2);
      const corr = denom > 0 ? cov / denom : 0;
      nacf[period] = corr;
      if (corr > globalMax) globalMax = corr;
    }

    // 取第一个显著峰：谐波信号在基频周期的整数倍处都会出现强相关，
    // 最短的那个周期才对应真正的基频
    const threshold = Math.min(0.85, 0.97 * globalMax);
    for (let period = minPeriod + 1; period < maxPeriod - 1; period++) {
      if (
        nacf[period] >= threshold &&
        nacf[period] >= nacf[period - 1] &&
        nacf[period] > nacf[period + 1]
      ) {
        // 抛物线插值获得亚采样级精度的周期
        const y0 = nacf[period - 1];
        const y1 = nacf[period];
        const y2 = nacf[period + 1];
        const denom = y0 - 2 * y1 + y2;
        const shift = denom !== 0 ? 0.5 * (y0 - y2) / denom : 0;
        const refinedPeriod = period + Math.max(-1, Math.min(1, shift));
        return sampleRate / refinedPeriod;
      }
    }

    // 兜底：找不到显著峰时返回全局最大滞后
    let bestPeriod = minPeriod;
    let bestCorr = -Infinity;
    for (let period = minPeriod; period < maxPeriod; period++) {
      if (nacf[period] > bestCorr) {
        bestCorr = nacf[period];
        bestPeriod = period;
      }
    }
    return sampleRate / bestPeriod;
  }

  /**
   * 峰值检测法
   */
  findDominantPeak(frequencies, magnitudes) {
    let maxMag = 0;
    let peakFreq = 100;

    // 在合理的基频范围内寻找最大峰值 (古琴基频通常在 60-500Hz)
    for (let i = 0; i < frequencies.length; i++) {
      if (frequencies[i] >= 50 && frequencies[i] <= 1000) {
        if (magnitudes[i] > maxMag) {
          maxMag = magnitudes[i];
          peakFreq = frequencies[i];
        }
      }
    }

    return peakFreq;
  }

  /**
   * 倍频/半频校验
   *
   * 基频候选位置若几乎没有能量（低于全频段最大能量的 0.5%），
   * 而其 2 倍频处能量显著（至少高一个数量级），说明自相关把
   * 2 倍周期当成了周期，即结果低了一个八度——向上修正。
   * 循环检查可同时应对低两个八度的极端误检。
   */
  correctOctaveError(freq, frequencies, magnitudes) {
    if (!frequencies.length) return freq;

    const globalMax = Math.max(...magnitudes);
    if (globalMax <= 0) return freq;

    const energyAt = (targetFreq) => {
      // 容差取 3% 与一个频率分辨率中的较大者
      let binWidth = Infinity;
      for (let i = 1; i < frequencies.length; i++) {
        binWidth = Math.min(binWidth, frequencies[i] - frequencies[i - 1]);
      }
      const tolerance = Math.max(targetFreq * 0.03, binWidth);
      let energy = 0;
      for (let i = 0; i < frequencies.length; i++) {
        if (Math.abs(frequencies[i] - targetFreq) <= tolerance) {
          energy = Math.max(energy, magnitudes[i]);
        }
      }
      return energy;
    };

    let candidate = freq;
    for (let octave = 0; octave < 2; octave++) {
      if (candidate * 2 > 1000) break;
      const selfEnergy = energyAt(candidate);
      const doubleEnergy = energyAt(candidate * 2);
      if (
        selfEnergy < 0.005 * globalMax &&
        doubleEnergy > 0.01 * globalMax &&
        doubleEnergy > 10 * selfEnergy
      ) {
        logger.info('检测到低八度误检，基频上移一个八度', {
          from: candidate,
          to: candidate * 2
        });
        candidate *= 2;
      } else {
        break;
      }
    }

    return candidate;
  }

  /**
   * 计算倍频
   */
  calculateHarmonics(fundamentalFreq, maxHarmonic = 13) {
    const harmonics = [];
    for (let i = 2; i <= maxHarmonic; i++) {
      harmonics.push(fundamentalFreq * i);
    }
    return harmonics;
  }

  /**
   * 过滤只保留基频和倍频的数据
   */
  filterHarmonics(frequencies, magnitudes, fundamentalFreq, harmonics) {
    const allHarmonics = [fundamentalFreq, ...harmonics];
    const filteredFreqs = [];
    const filteredMags = [];
    const tolerance = fundamentalFreq * 0.1; // 10% 容差
    
    for (let i = 0; i < frequencies.length; i++) {
      const freq = frequencies[i];
      
      // 检查是否接近任何一个谐波
      for (const harmonic of allHarmonics) {
        if (Math.abs(freq - harmonic) < tolerance) {
          filteredFreqs.push(freq);
          filteredMags.push(magnitudes[i]);
          break;
        }
      }
    }
    
    return { frequencies: filteredFreqs, magnitudes: filteredMags };
  }

  /**
   * 计算频率区域数据
   */
  calculateFrequencyBands(fundamentalFreq, harmonics, filteredData) {
    const allHarmonics = [fundamentalFreq, ...harmonics];
    
    // 低频区: 基频 ~ 4倍频
    const lowFreqRange = { min: fundamentalFreq * 0.9, max: fundamentalFreq * 4.5 };
    // 中频区: 5倍频 ~ 8倍频
    const midFreqRange = { min: fundamentalFreq * 4.5, max: fundamentalFreq * 8.5 };
    // 高频区: 9倍频 ~ 13倍频
    const highFreqRange = { min: fundamentalFreq * 8.5, max: fundamentalFreq * 13.5 };

    const extractBandData = (range) => {
      const freqs = [];
      const mags = [];
      
      for (let i = 0; i < filteredData.frequencies.length; i++) {
        const freq = filteredData.frequencies[i];
        if (freq >= range.min && freq <= range.max) {
          freqs.push(freq);
          mags.push(filteredData.magnitudes[i]);
        }
      }
      
      // 为每个倍频创建数据点
      const bandHarmonics = allHarmonics.filter(h => h >= range.min && h <= range.max);
      const harmonicData = bandHarmonics.map(h => {
        // 找到最接近的实际数据点
        let closestMag = 0;
        let minDist = Infinity;
        
        for (let i = 0; i < freqs.length; i++) {
          const dist = Math.abs(freqs[i] - h);
          if (dist < minDist) {
            minDist = dist;
            closestMag = mags[i];
          }
        }
        
        return { frequency: h, magnitude: closestMag };
      });
      
      return harmonicData;
    };

    return {
      low: extractBandData(lowFreqRange),
      mid: extractBandData(midFreqRange),
      high: extractBandData(highFreqRange)
    };
  }

  /**
   * 计算热力图数据 - 声强随时间变化
   */
  calculateHeatmapData(audioData, sampleRate, fftSize, fundamentalFreq, harmonics) {
    const allHarmonics = [fundamentalFreq, ...harmonics];
    const windowSize = Math.min(fftSize, 2048);
    const hopSize = windowSize / 4;
    const numFrames = Math.floor((audioData.length - windowSize) / hopSize) + 1;
    
    // 限制帧数以提高性能
    const maxFrames = 100;
    const frameStep = Math.max(1, Math.floor(numFrames / maxFrames));
    const actualFrames = Math.ceil(numFrames / frameStep);
    
    const heatmapData = [];
    const timeLabels = [];
    const freqLabels = allHarmonics.map((h, i) => i === 0 ? '基频' : `${i + 1}倍频`);
    
    for (let frame = 0; frame < numFrames; frame += frameStep) {
      const startSample = frame * hopSize;
      const endSample = startSample + windowSize;
      
      if (endSample > audioData.length) break;
      
      const frameData = audioData.slice(startSample, endSample);
      const fftResult = this.performFFT(frameData, windowSize);
      const freqResolution = sampleRate / windowSize;
      
      // 提取每个谐波的能量
      const frameEnergies = allHarmonics.map(harmonic => {
        const binIndex = Math.round(harmonic / freqResolution);
        if (binIndex >= 0 && binIndex < fftResult.length) {
          return fftResult[binIndex];
        }
        return 0;
      });
      
      heatmapData.push(frameEnergies);
      timeLabels.push((startSample / sampleRate * 1000).toFixed(0));
    }
    
    // 归一化
    const maxVal = Math.max(...heatmapData.flat());
    const normalizedData = heatmapData.map(row => 
      row.map(val => maxVal > 0 ? val / maxVal : 0)
    );
    
    return {
      data: normalizedData,
      timeLabels,
      freqLabels
    };
  }
}
