// routes.swift

import CoreGraphics
import FluxSwift
import Foundation
import Hub
import ImageIO
import MLX
import MLXNN
import MLXRandom
import Progress
import Tokenizers
import UniformTypeIdentifiers
import Vapor

// MARK: - CORS Configuration
public func configureCORS(_ app: Application) throws {
    // Configure CORS middleware
    let corsConfiguration = CORSMiddleware.Configuration(
        allowedOrigin: .any(["http://localhost:5173"]),
        allowedMethods: [.GET, .POST, .OPTIONS, .DELETE, .PATCH],
        allowedHeaders: [
            .accept,
            .authorization,
            .contentType,
            .origin,
            .xRequestedWith,
            .userAgent,
            .accessControlAllowOrigin
        ]
    )
    
    let cors = CORSMiddleware(configuration: corsConfiguration)
    
    // Use the CORS middleware
    app.middleware.use(cors)
}

// MARK: - Models

struct GenerateRequest: Content {
    var prompt: String?
    var width: Int?
    var height: Int?
    var steps: Int?
    var guidance: Float?
    var output: String?
    var repo: String?
    var seed: UInt64?
    var quantize: Bool?
    var float16: Bool?
    var model: String?
    var hfToken: String?
    var loraPath: String?
    var initImagePath: String?
    var initImageBase64: String?
    var initImageStrength: Float?
    
    // Helper properties with validation
    var finalWidth: Int {
        let w = width ?? 512
        return w - (w % 64) // Ensure it's a multiple of 64
    }
    
    var finalHeight: Int {
        let h = height ?? 512
        return h - (h % 64) // Ensure it's a multiple of 64
    }
    
    var finalSteps: Int {
        min(max(steps ?? 4, 1), 50) // Clamp between 1 and 50
    }
    
    var finalGuidance: Float {
        min(max(guidance ?? 3.5, 0.0), 10.0) // Clamp between 0 and 10
    }
    
    var finalImageStrength: Float {
        min(max(initImageStrength ?? 0.3, 0.0), 1.0) // Clamp between 0 and 1
    }
    
    var finalModel: String {
        return model?.lowercased() ?? "schnell" // Default is schnell if not passed
    }
    
    var finalFloat16: Bool {
        return float16 ?? true // Default is true if not passed
    }
    
    var finalQuantize: Bool {
        return quantize ?? false // Default is false if not passed
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
    let model: String?
}

// MARK: - Job Status

struct JobStatus {
    enum Status {
        case inProgress(progress: Int)
        case completed(imagePath: String)
        case failed(error: String)
    }
    
    let status: Status
    let request: GenerateRequest
}

// MARK: - Job Storage

actor JobStorage {
    private var jobs: [String: JobStatus] = [:]
    
    func createJob(id: String, status: JobStatus.Status, request: GenerateRequest) {
        jobs[id] = JobStatus(status: status, request: request)
    }
    
    func updateJob(id: String, status: JobStatus.Status, request: GenerateRequest) {
        jobs[id] = JobStatus(status: status, request: request)
    }
    
    func getJob(id: String) -> JobStatus? {
        return jobs[id]
    }
    
    func cleanupJob(id: String) {
        jobs.removeValue(forKey: id)
    }
    
    func deleteJob(id: String) {
        jobs.removeValue(forKey: id)
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

// Function to select the correct FluxConfiguration
func selectFluxConfiguration(modelType: String, hfToken: String?) -> FluxConfiguration {
    switch modelType {
    case "dev":
        let token = hfToken
        return FluxConfiguration.flux1Dev
    case "schnell":
        return FluxConfiguration.flux1Schnell
    default:
        return FluxConfiguration.flux1Schnell // Default to schnell
    }
}

// Global job storage
let jobStorage = JobStorage()

// MARK: - Routes

func routes(_ app: Application) throws {
    app.routes.defaultMaxBodySize = "50mb"
    
    // Configure CORS first
    let corsConfiguration = CORSMiddleware.Configuration(
        allowedOrigin: .any(["http://localhost:5173"]),
        allowedMethods: [.GET, .POST, .OPTIONS, .DELETE, .PATCH],
        allowedHeaders: [
            .accept,
            .authorization,
            .contentType,
            .origin,
            .xRequestedWith,
            .userAgent,
            .accessControlAllowOrigin
        ]
    )
    app.middleware.use(CORSMiddleware(configuration: corsConfiguration))
    
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
        var request = try req.content.decode(GenerateRequest.self)
        let jobId = UUID().uuidString
        
        // Handle base64 image if provided
        if let base64Str = request.initImageBase64 {
            // Strip data URI prefix if present
            let dataURIPrefixPattern = "^data:image\\/\\w+;base64,"
            if let range = base64Str.range(of: dataURIPrefixPattern, options: .regularExpression) {
                request.initImageBase64 = String(base64Str[range.upperBound...])
            }
            
            // Decode base64 and save to temp file
            if let imageData = Data(base64Encoded: request.initImageBase64 ?? "") {
                let tmpInitFileName = "tmp_\(jobId).png"
                let tmpInitFullPath = imagesPath + tmpInitFileName
                
                try imageData.write(to: URL(fileURLWithPath: tmpInitFullPath))
                request.initImagePath = tmpInitFullPath
                req.logger.info("Received base64 init image; wrote to \(tmpInitFullPath)")
            }
        }
        
        req.logger.info("Received request: prompt='\(request.prompt ?? "default prompt")', width=\(request.finalWidth), height=\(request.finalHeight), steps=\(request.finalSteps), model=\(request.finalModel)")
        
        // Create job entry
        await jobStorage.createJob(id: jobId, status: .inProgress(progress: 0), request: request)
        req.logger.info("Created job with ID: \(jobId)")
        
        Task {
            do {
                // Set up model configuration
                let selectedModel = selectFluxConfiguration(modelType: request.finalModel, hfToken: request.hfToken)
                
                // Download model
                req.logger.info("Downloading or loading model...")
                
                let hubApi = request.finalModel == "dev" ? HubApi(hfToken: request.hfToken) : HubApi()

                try await selectedModel.download(hub: hubApi)
                
                let loadConfiguration = LoadConfiguration(
                    float16: request.finalFloat16,
                    quantize: request.finalQuantize,
                    loraPath: request.loraPath
                )

                if let loraPath = loadConfiguration.loraPath {
                    if !FileManager.default.fileExists(atPath: loraPath) {
                        try await selectedModel.downloadLoraWeights(loadConfiguration: loadConfiguration) {
                            _ in
                      
                            req.logger.info("Downloading lora weights for \(loraPath) model...")
                        }
                    }
                }

                // Set up generator and parameters
                let parameters = EvaluateParameters(
                    width: request.finalWidth,
                    height: request.finalHeight,
                    numInferenceSteps: request.finalSteps,
                    guidance: request.finalGuidance,
                    seed: request.seed,
                    prompt: request.prompt ?? "default prompt",
                    shiftSigmas: request.finalModel == "dev" ? true : false
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
                                    
                    // Load and preprocess the input image
                    let sourceImage = try Image(url: URL(fileURLWithPath: initImagePath))

                    // Ensure exact dimensions by resizing.
                    let resizedImage = Image(image: sourceImage.asCGImage(), maximumEdge: max(request.finalWidth, request.finalHeight))
                    
                    // Ensure exact dimensions
                    let targetWidth = request.finalWidth
                    let targetHeight = request.finalHeight
                    
                    // Log the actual dimensions we got
                    let (H, W, C) = resizedImage.data.shape3
                    req.logger.info("Resized image dimensions: \(W)x\(H) with \(C) channels")
                                    
                    // Convert to float and normalize
                    let input = (resizedImage.data.asType(.float32) / 255) * 2 - 1
                                    
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
                        await jobStorage.updateJob(
                            id: jobId,
                            status: .inProgress(progress: progress),
                            request: request
                        )
                        req.logger.info("Updated job \(jobId) to progress \(progress)")
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
                        req.logger.info("Updated job \(jobId) to progress \(progress)")
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
                
                // Clean up temporary init image after successful generation
                if let tmpPath = request.initImagePath,
                   tmpPath.contains("tmp_\(jobId).png"),
                   FileManager.default.fileExists(atPath: tmpPath)
                {
                    try? FileManager.default.removeItem(atPath: tmpPath)
                    req.logger.info("Cleaned up temporary init image at \(tmpPath)")
                }
                
            } catch {
                // Clean up temporary file in case of error too
                if let tmpPath = request.initImagePath,
                   tmpPath.contains("tmp_\(jobId).png"),
                   FileManager.default.fileExists(atPath: tmpPath)
                {
                    try? FileManager.default.removeItem(atPath: tmpPath)
                    req.logger.info("Cleaned up temporary init image after error at \(tmpPath)")
                }
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
        
        req.logger.info("Requesting status for job: \(jobId)")
        
        guard let job = await jobStorage.getJob(id: jobId) else {
            req.logger.info("Job \(jobId) not found in status check.")
            throw Abort(.notFound, reason: "Job not found")
        }
        req.logger.info("Job \(jobId) status: \(job.status)")

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
                steps: job.request.finalSteps,
                model: job.request.finalModel
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
                steps: job.request.finalSteps,
                model: job.request.finalModel
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
                steps: job.request.finalSteps,
                model: job.request.finalModel
            )
        }
    }
    
    // MARK: Download endpoint
    
    app.get("download", ":jobId") { req async throws -> Response in
        guard let jobId = req.parameters.get("jobId") else {
            throw Abort(.badRequest, reason: "Job ID is required")
        }
        req.logger.info("Requesting download for job: \(jobId)")
        
        guard let job = await jobStorage.getJob(id: jobId) else {
            req.logger.info("Job \(jobId) not found in download check.")
            throw Abort(.notFound, reason: "Image not found or job not completed")
        }
        
        guard case .completed(let imagePath) = job.status else {
            throw Abort(.notFound, reason: "Image not found or job not completed")
        }
        
        req.logger.info("Job \(jobId) status at download check: \(job.status)")

        let fullPath = publicPath + imagePath
        guard FileManager.default.fileExists(atPath: fullPath) else {
            throw Abort(.notFound, reason: "Image file not found")
        }
        
        let data = try Data(contentsOf: URL(fileURLWithPath: fullPath))
        
        let response = Response(status: .ok)
        response.headers.contentType = .png
        response.body = .init(data: data)
        
        // Clean the job only after the file was delivered.
        await jobStorage.deleteJob(id: jobId)
        try? FileManager.default.removeItem(atPath: fullPath)
        
        return response
    }
}
