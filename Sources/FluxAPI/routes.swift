// routes.swift
import Vapor
import FluxSwift
import Foundation
import Hub
import MLX
import MLXNN
import MLXRandom
import Progress
import Tokenizers
import CoreGraphics
import UniformTypeIdentifiers
import ImageIO

// MARK: - Models
struct GenerateImageRequest: Content {
    var prompt: String
    var width: Int?
    var height: Int?
    var steps: Int?
    var initImagePath: String?
    var initImageStrength: Float?
    
    // Helper properties with validation
    var finalWidth: Int {
        let w = width ?? 512
        return w - (w % 64)  // Ensure it's a multiple of 64
    }
    
    var finalHeight: Int {
        let h = height ?? 512
        return h - (h % 64)  // Ensure it's a multiple of 64
    }
    
    var finalSteps: Int {
        min(max(steps ?? 4, 1), 50)  // Clamp between 1 and 50
    }
    
    var finalImageStrength: Float {
        min(max(initImageStrength ?? 0.3, 0.0), 1.0)  // Clamp between 0 and 1
    }
}

struct JobResponse: Content {
    let jobId: String
}

struct JobStatusResponse: Content {
    let status: String
    let progress: Int
    let imagePath: String?
    let error: String?
    let prompt: String?
    let width: Int?
    let height: Int?
    let steps: Int?
}

// MARK: - Job Status
struct JobStatus {
    enum Status {
        case inProgress(progress: Int)
        case completed(imagePath: String)
        case failed(error: String)
    }
    
    let status: Status
    let request: GenerateImageRequest
}

// MARK: - Job Storage
actor JobStorage {
    private var jobs: [String: JobStatus] = [:]
    
    func createJob(id: String, status: JobStatus.Status, request: GenerateImageRequest) {
        jobs[id] = JobStatus(status: status, request: request)
    }
    
    func updateJob(id: String, status: JobStatus.Status, request: GenerateImageRequest) {
        jobs[id] = JobStatus(status: status, request: request)
    }
    
    func getJob(id: String) -> JobStatus? {
        return jobs[id]
    }
    
    func cleanupJob(id: String) {
        jobs.removeValue(forKey: id)
    }
}

// MARK: - Image Processing
enum ImageError: Error {
    case failedToSave
    case unableToOpen
}

public struct Image {
    public let data: MLXArray

    public init(_ data: MLXArray) {
        precondition(data.ndim == 3)
        self.data = data
    }

    public init(url: URL, maximumEdge: Int? = nil) throws {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw ImageError.unableToOpen
        }
        self.init(image: image)
    }

    public init(image: CGImage, maximumEdge: Int? = nil) {
        var width = image.width
        var height = image.height

        if let maximumEdge {
            func scale(_ edge: Int, _ maxEdge: Int) -> Int {
                Int(round(Float(maximumEdge) / Float(maxEdge) * Float(edge)))
            }

            if width >= height {
                width = scale(width, image.width)
                height = scale(height, image.width)
            } else {
                width = scale(width, image.height)
                height = scale(height, image.height)
            }
        }

        width = width - width % 64
        height = height - height % 64

        var raster = Data(count: width * 4 * height)
        raster.withUnsafeMutableBytes { ptr in
            let cs = CGColorSpace(name: CGColorSpace.sRGB)!
            let context = CGContext(
                data: ptr.baseAddress, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: width * 4, space: cs,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
                    | CGBitmapInfo.byteOrder32Big.rawValue)!

            context.draw(
                image, in: CGRect(origin: .zero, size: .init(width: width, height: height)))
        }

        self.data = MLXArray(raster, [height, width, 4], type: UInt8.self)[0..., 0..., ..<3]
    }

    public func asCGImage() -> CGImage {
        var raster = data

        if data.dim(-1) == 3 {
            raster = padded(raster, widths: [0, 0, [0, 1]])
        }

        class DataHolder {
            var data: Data
            init(_ data: Data) {
                self.data = data
            }
        }

        let holder = DataHolder(raster.asData())
        let payload = Unmanaged.passRetained(holder).toOpaque()
        
        func release(payload: UnsafeMutableRawPointer?, data: UnsafeMutableRawPointer?) {
            Unmanaged<DataHolder>.fromOpaque(payload!).release()
        }

        return holder.data.withUnsafeMutableBytes { ptr in
            let (H, W, C) = raster.shape3
            let cs = CGColorSpace(name: CGColorSpace.sRGB)!

            let context = CGContext(
                data: ptr.baseAddress, width: W, height: H, bitsPerComponent: 8, bytesPerRow: W * C,
                space: cs,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
                    | CGBitmapInfo.byteOrder32Big.rawValue, releaseCallback: release,
                releaseInfo: payload)!
            return context.makeImage()!
        }
    }

    public func save(url: URL) throws {
        let uti = UTType(filenameExtension: url.pathExtension) ?? UTType.png
        
        guard let destination = CGImageDestinationCreateWithURL(
            url as CFURL, uti.identifier as CFString, 1, nil)
        else {
            throw ImageError.failedToSave
        }
        
        CGImageDestinationAddImage(destination, asCGImage(), nil)
        if !CGImageDestinationFinalize(destination) {
            throw ImageError.failedToSave
        }
    }
}

// MARK: - Helper Functions
func calculateProgress(currentStep: Int, totalSteps: Int) -> Int {
    switch currentStep {
        case 0: return 0
        case 1: return 25
        case 2: return 50
        case 3: return 75
        case totalSteps: return 100
        default: return min(Int((Float(currentStep) / Float(totalSteps)) * 100), 100)
    }
}

private func unpackLatents(_ latents: MLXArray, height: Int, width: Int) -> MLXArray {
    let reshaped = latents.reshaped(1, height / 16, width / 16, 16, 2, 2)
    let transposed = reshaped.transposed(0, 1, 4, 2, 5, 3)
    return transposed.reshaped(1, height / 16 * 2, width / 16 * 2, 16)
}

// Global job storage
let jobStorage = JobStorage()

// MARK: - Routes
func routes(_ app: Application) throws {
    // Get the absolute path to the Public directory
    let workingDirectory = DirectoryConfiguration.detect().workingDirectory
    let publicPath = workingDirectory + "Public/"
    let imagesPath = publicPath + "images/"
    
    // Ensure images directory exists
    let fileManager = FileManager.default
    if !fileManager.fileExists(atPath: imagesPath) {
        try fileManager.createDirectory(atPath: imagesPath, withIntermediateDirectories: true)
    }
    
    app.logger.info("Images will be saved to: \(imagesPath)")
    
    // MARK: Generate endpoint
    app.post("generate") { req async throws -> JobResponse in
        let request = try req.content.decode(GenerateImageRequest.self)
        let jobId = UUID().uuidString
        
        req.logger.info("Received request: prompt='\(request.prompt)', width=\(request.finalWidth), height=\(request.finalHeight), steps=\(request.finalSteps)")
        
        // Create job entry
        await jobStorage.createJob(id: jobId, status: .inProgress(progress: 0), request: request)
        
        Task {
            do {
                // Set up model configuration
                let selectedModel = FluxConfiguration.flux1Schnell
                
                // Download model
                req.logger.info("Downloading or loading model...")
                try await selectedModel.download(hub: HubApi())
                
                let loadConfiguration = LoadConfiguration(
                    float16: true,
                    quantize: false,
                    loraPath: nil
                )
                
                // Set up generator and parameters
                let parameters = EvaluateParameters(
                                    width: request.finalWidth,
                                    height: request.finalHeight,
                                    numInferenceSteps: request.finalSteps,
                                    guidance: 3.5,
                                    seed: nil,
                                    prompt: request.prompt,
                                    shiftSigmas: false
                                )
                                
                                req.logger.info("Initializing generator...")
                                
                                // Variable to store the final image
                                let generatedImage: Image
                                
                if let initImagePath = request.initImagePath {
                                    // Image-to-image generation
                                    req.logger.info("Using image-to-image generation with reference image: \(initImagePath)")
                                    guard var generator = try selectedModel.ImageToImageGenerator(configuration: loadConfiguration) else {
                                        throw Abort(.internalServerError, reason: "Failed to create image-to-image generator")
                                    }
                                    
                                    generator.ensureLoaded()
                                    
                                    // Ensure exact dimensions
                                    let targetWidth = request.finalWidth
                                    let targetHeight = request.finalHeight
                                    
                                    req.logger.info("Processing input image to exact dimensions: \(targetWidth)x\(targetHeight)")
                                    
                                    // Create a new Image struct for resizing
                                    let sourceImage = try Image(url: URL(fileURLWithPath: initImagePath))
                                    
                                    // Create a context with exact dimensions
                                    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
                                    var resizeRaster = Data(count: targetWidth * 4 * targetHeight)
                                    resizeRaster.withUnsafeMutableBytes { ptr in
                                        let context = CGContext(
                                            data: ptr.baseAddress,
                                            width: targetWidth,
                                            height: targetHeight,
                                            bitsPerComponent: 8,
                                            bytesPerRow: targetWidth * 4,
                                            space: cs,
                                            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
                                        )!
                                        
                                        // Draw the source image scaled to fit exactly
                                        context.draw(
                                            sourceImage.asCGImage(),
                                            in: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight)
                                        )
                                    }
                                    
                                    // Create MLXArray with exact dimensions
                                    let resizedData = MLXArray(resizeRaster, [targetHeight, targetWidth, 4], type: UInt8.self)[0..., 0..., ..<3]
                                    let image = Image(resizedData)
                                    
                                    // Log the actual dimensions we got
                                    let (H, W, C) = image.data.shape3
                                    req.logger.info("Resized image dimensions: \(W)x\(H) with \(C) channels")
                                    
                                    // Convert to float and normalize
                                    let input = (image.data.asType(.float32) / 255) * 2 - 1
                                    
                                    let strength = request.finalImageStrength
                                    req.logger.info("Using image strength: \(strength)")
                                    
                                    guard var denoiser = (generator as? ImageToImageGenerator)?.generateLatents(
                                        image: input,
                                        parameters: parameters,
                                        strength: strength
                                    ) else {
                                        throw Abort(.internalServerError, reason: "Failed to create image-to-image denoiser")
                                    }
                                    
                                    // Generate image
                                    req.logger.info("Starting image-to-image generation...")
                                    var step = 0
                                    var lastXt: MLXArray!
                                    
                                    while let xt = denoiser.next() {
                                        step += 1
                                        let progress = calculateProgress(currentStep: step, totalSteps: parameters.numInferenceSteps)
                                        await jobStorage.updateJob(
                                            id: jobId,
                                            status: .inProgress(progress: progress),
                                            request: request
                                        )
                                        req.logger.info("Step \(step)/\(parameters.numInferenceSteps) - Progress: \(progress)%")
                                        eval(xt)
                                        lastXt = xt
                                    }
                                    
                                    // Process generated image
                                    req.logger.info("Processing generated image...")
                                    let unpackedLatents = unpackLatents(lastXt, height: parameters.height, width: parameters.width)
                                    let decoded = generator.decode(xt: unpackedLatents)
                                    let imageData = decoded.squeezed()
                                    let raster = (imageData * 255).asType(.uint8)
                                    
                                    // Create Image instance
                                    generatedImage = Image(raster)
                                    
                                } else {
                                    // Text-to-image generation (existing code)
                                    guard var generator = try selectedModel.textToImageGenerator(configuration: loadConfiguration) else {
                                        throw Abort(.internalServerError, reason: "Failed to create generator")
                                    }
                                    
                                    generator.ensureLoaded()
                                    guard var denoiser = (generator as? TextToImageGenerator)?.generateLatents(parameters: parameters) else {
                                        throw Abort(.internalServerError, reason: "Failed to create denoiser")
                                    }
                                    
                                    // Generate image
                                    req.logger.info("Starting text-to-image generation...")
                                    var step = 0
                                    var lastXt: MLXArray!
                                    
                                    while let xt = denoiser.next() {
                                        step += 1
                                        let progress = calculateProgress(currentStep: step, totalSteps: parameters.numInferenceSteps)
                                        await jobStorage.updateJob(
                                            id: jobId,
                                            status: .inProgress(progress: progress),
                                            request: request
                                        )
                                        req.logger.info("Step \(step)/\(parameters.numInferenceSteps) - Progress: \(progress)%")
                                        eval(xt)
                                        lastXt = xt
                                    }
                                    
                                    // Process generated image
                                    req.logger.info("Processing generated image...")
                                    let unpackedLatents = unpackLatents(lastXt, height: parameters.height, width: parameters.width)
                                    let decoded = generator.decode(xt: unpackedLatents)
                                    let imageData = decoded.squeezed()
                                    let raster = (imageData * 255).asType(.uint8)
                                    
                                    // Create Image instance
                                    generatedImage = Image(raster)
                                }
                                
                                // Save the generated image
                                let imageName = jobId + ".png"
                                let imagePath = imagesPath + imageName
                                
                                req.logger.info("Saving image to: \(imagePath)")
                                try generatedImage.save(url: URL(fileURLWithPath: imagePath))
                                
                                req.logger.info("Image saved successfully")
                                await jobStorage.updateJob(
                                    id: jobId,
                                    status: .completed(imagePath: "images/" + imageName),
                                    request: request
                                )
                
                // Schedule cleanup
                Task {
                    try await Task.sleep(nanoseconds: 3600_000_000_000) // 1 hour
                    try? FileManager.default.removeItem(atPath: imagePath)
                    await jobStorage.cleanupJob(id: jobId)
                }
                
            } catch {
                req.logger.error("Error generating image: \(error)")
                await jobStorage.updateJob(
                    id: jobId,
                    status: .failed(error: error.localizedDescription),
                    request: request
                )
            }
        }
        
        return JobResponse(jobId: jobId)
    }
    
    // MARK: Status endpoint
    app.get("status", ":jobId") { req async throws -> JobStatusResponse in
        guard let jobId = req.parameters.get("jobId") else {
            throw Abort(.badRequest, reason: "Job ID is required")
        }
        
        guard let job = await jobStorage.getJob(id: jobId) else {
            throw Abort(.notFound, reason: "Job not found")
        }
        
        switch job.status {
        case .inProgress(let progress):
            return JobStatusResponse(
                status: "in_progress",
                progress: progress,
                imagePath: nil,
                error: nil,
                prompt: job.request.prompt,
                width: job.request.finalWidth,
                height: job.request.finalHeight,
                steps: job.request.finalSteps
            )
        case .completed(let imagePath):
            return JobStatusResponse(
                status: "completed",
                progress: 100,
                imagePath: imagePath,
                error: nil,
                prompt: job.request.prompt,
                width: job.request.finalWidth,
                height: job.request.finalHeight,
                steps: job.request.finalSteps
            )
        case .failed(let error):
            return JobStatusResponse(
                status: "failed",
                progress: 0,
                imagePath: nil,
                error: error,
                prompt: job.request.prompt,
                width: job.request.finalWidth,
                height: job.request.finalHeight,
                steps: job.request.finalSteps
            )
        }
    }
    
    // MARK: Download endpoint
    app.get("download", ":jobId") { req -> Response in
        guard let jobId = req.parameters.get("jobId") else {
            throw Abort(.badRequest, reason: "Job ID is required")
        }
        
        let status = await jobStorage.getJob(id: jobId)
        guard case .completed(let imagePath) = status?.status else {
            throw Abort(.notFound, reason: "Image not found or job not completed")
        }
        
        let fullPath = publicPath + imagePath
        guard FileManager.default.fileExists(atPath: fullPath) else {
            throw Abort(.notFound, reason: "Image file not found")
        }
        
        let data = try Data(contentsOf: URL(fileURLWithPath: fullPath))
        
        let response = Response(status: .ok)
        response.headers.contentType = .png
        response.body = .init(data: data)
        return response
    }
}
