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
    
    // 应用汉宁窗
    for (let i = 0; i < copyLength; i++) {
      const window = 0.5 * (1 - Math.cos(2 * Math.PI * i / (copyLength - 1)));
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

    // 位反转排序
    const real = new Float32Array(n);
    const imag = new Float32Array(n);
    const levels = Math.round(Math.log2(n));

    for (let i = 0; i < n; i++) {
      let reversed = 0;
      for (let bit = 0; bit < levels; bit++) {
        reversed = (reversed << 1) | ((i >>> bit) & 1);
      }
      real[reversed] = data[i];
      imag[reversed] = 0;
    }

    // 迭代 FFT
    for (let size = 2; size <= n; size *= 2) {
      const halfSize = size / 2;
      const step = n / size;
      
      for (let i = 0; i < n; i += size) {
        for (let j = 0; j < halfSize; j++) {
          const angle = -2 * Math.PI * j * step / n;
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
   * 检测基频 - 使用自相关法和峰值检测
   */
  detectFundamentalFrequency(audioData, sampleRate, frequencies, magnitudes) {
    // 方法1: 自相关法
    const autocorrFreq = this.autocorrelation(audioData, sampleRate);
    
    // 方法2: 峰值检测法
    const peakFreq = this.findDominantPeak(frequencies, magnitudes);
    
    // 综合判断 - 优先使用自相关法的结果，因为它对古琴这类乐器更准确
    let fundamentalFreq = autocorrFreq;
    
    // 如果自相关法结果不合理，使用峰值检测
    if (fundamentalFreq < 50 || fundamentalFreq > 2000) {
      fundamentalFreq = peakFreq;
    }
    
    // 验证：检查是否可能是倍频被误检为基频
    const possibleFundamental = this.verifyFundamental(fundamentalFreq, frequencies, magnitudes);
    
    logger.info('基频检测结果', { autocorrFreq, peakFreq, final: possibleFundamental });
    
    return possibleFundamental;
  }

  /**
   * 自相关法检测基频
   */
  autocorrelation(audioData, sampleRate) {
    const minPeriod = Math.floor(sampleRate / 2000); // 最高频率 2000Hz
    const maxPeriod = Math.floor(sampleRate / 50);   // 最低频率 50Hz
    const dataLength = Math.min(audioData.length, sampleRate); // 最多分析1秒
    const periodLimit = Math.min(maxPeriod, Math.floor(dataLength / 2));

    // 计算所有候选周期的相关值
    const correlations = new Float32Array(periodLimit + 1);
    let maxCorr = 0;
    let maxCorrPeriod = minPeriod;

    for (let period = minPeriod; period < periodLimit; period++) {
      let corr = 0;
      let count = 0;

      for (let i = 0; i < dataLength - period; i++) {
        corr += audioData[i] * audioData[i + period];
        count++;
      }

      corr /= count;
      correlations[period] = corr;

      if (corr > maxCorr) {
        maxCorr = corr;
        maxCorrPeriod = period;
      }
    }

    // 周期信号的自相关在 T、2T、3T... 处都会出现峰值，
    // 直接取全局最大值容易把 2T 误判为基频（低八度误检）。
    // 改为从最短周期开始，取第一个接近全局最大值的局部峰值。
    const threshold = maxCorr * 0.95;
    let bestPeriod = maxCorrPeriod;

    for (let period = minPeriod + 1; period < periodLimit - 1; period++) {
      const corr = correlations[period];
      if (corr >= threshold &&
          corr >= correlations[period - 1] &&
          corr >= correlations[period + 1]) {
        bestPeriod = period;
        break;
      }
    }

    // 抛物线插值提高周期精度
    let refinedPeriod = bestPeriod;
    if (bestPeriod > minPeriod && bestPeriod < periodLimit - 1) {
      const y1 = correlations[bestPeriod - 1];
      const y2 = correlations[bestPeriod];
      const y3 = correlations[bestPeriod + 1];
      const denominator = y1 - 2 * y2 + y3;
      if (denominator !== 0) {
        refinedPeriod = bestPeriod + 0.5 * (y1 - y3) / denominator;
      }
    }

    return sampleRate / refinedPeriod;
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
   * 验证基频 - 检查是否把倍频或次谐波误检为基频
   */
  verifyFundamental(freq, frequencies, magnitudes) {
    if (magnitudes.length === 0) return freq;

    const maxMagnitude = Math.max(...magnitudes);
    if (maxMagnitude <= 0) return freq;

    // 查找目标频率附近的最强能量
    const energyNear = (targetFreq) => {
      const tolerance = targetFreq * 0.05; // 5% 容差
      let energy = 0;

      for (let i = 0; i < frequencies.length; i++) {
        if (Math.abs(frequencies[i] - targetFreq) < tolerance) {
          energy = Math.max(energy, magnitudes[i]);
        }
      }

      return energy;
    };

    // 向下检查：freq/2、freq/3 处若有显著能量，
    // 说明检测到的可能是倍频，真正的基频更低
    const possibleFundamentals = [freq / 2, freq / 3];

    for (const possibleFreq of possibleFundamentals) {
      if (possibleFreq < 50) continue;

      if (energyNear(possibleFreq) > 0.1 * maxMagnitude) {
        return possibleFreq;
      }
    }

    // 向上检查：若当前频率几乎没有能量，而其 2 倍频处能量显著，
    // 说明检测到的是次谐波（低八度误检），真正的基频更高
    if (energyNear(freq) < 0.05 * maxMagnitude) {
      const doubledFreq = freq * 2;
      if (doubledFreq <= 2000 && energyNear(doubledFreq) > 0.1 * maxMagnitude) {
        return doubledFreq;
      }
    }

    return freq;
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
