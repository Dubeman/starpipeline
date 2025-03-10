/** @type {import('next').NextConfig} */
const nextConfig = {
  // Configure API route options
  experimental: {
    serverActions: {
      bodySizeLimit: '32mb'
    }
  },
  // Increase response size limit
  api: {
    responseLimit: false,
    bodyParser: {
      sizeLimit: '32mb'
    }
  }
};

module.exports = nextConfig; 