/*
 * Copyright (c) 2014-2017 Christian Muehlhaeuser
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
 *	Authors:
 *		Christian Muehlhaeuser <muesli@gmail.com>
 *		Michael Wendland <michael@michiwend.com>
 *		Bjørn Erik Pedersen <bjorn.erik.pedersen@gmail.com>
 */

/*
Package smartcrop implements a content aware image cropping library based on
Jonas Wagner's smartcrop.js https://github.com/jwagner/smartcrop.js
*/
package smartcrop

import (
	"errors"
	"image"
	"image/color"
	"io/ioutil"
	"log"
	"math"
	"time"

	"github.com/muesli/smartcrop/options"

	"golang.org/x/image/draw"
)

var (
	// ErrInvalidDimensions gets returned when the supplied dimensions are invalid
	ErrInvalidDimensions = errors.New("Expect either a height or width")

	skinColor = [3]float64{0.78, 0.57, 0.44}
)

const (
	detailWeight            = 0.2
	skinBias                = 0.01
	skinBrightnessMin       = 0.2
	skinBrightnessMax       = 1.0
	skinThreshold           = 0.8
	skinWeight              = 1.8
	saturationBrightnessMin = 0.05
	saturationBrightnessMax = 0.9
	saturationThreshold     = 0.4
	saturationBias          = 0.2
	saturationWeight        = 0.3
	scoreDownSample         = 4 // step * minscale rounded down to the next power of two should be good
	step                    = 8
	scaleStep               = 0.1
	minScale                = 1.0
	maxScale                = 1.0
	edgeRadius              = 0.4
	edgeWeight              = -20.0
	outsideImportance       = -0.5
	boostWeight             = 200.0
	ruleOfThirds            = true
	prescale                = true
	prescaleMin             = 256.00
)

// Analyzer interface analyzes its struct and returns the best possible crop with the given
// width and height returns an error if invalid
type Analyzer interface {
	FindBestCrop(img image.Image, width, height int, boosts []BoostRegion) (image.Rectangle, error)
}

type BoostRegion struct {
	X 		int
	Y 		int
	Width 	int
	Height 	int
	Weight	float64
}

// Score contains values that classify matches
type Score struct {
	Detail     float64
	Saturation float64
	Skin       float64
	Boost      float64
}

// Crop contains results
type Crop struct {
	image.Rectangle
	Score Score
}

// Logger contains a logger.
type Logger struct {
	DebugMode bool
	Log       *log.Logger
}

type smartcropAnalyzer struct {
	logger Logger
	options.Resizer
}

// NewAnalyzer returns a new Analyzer using the given Resizer.
func NewAnalyzer(resizer options.Resizer) Analyzer {
	logger := Logger{
		DebugMode: false,
	}

	return NewAnalyzerWithLogger(resizer, logger)
}

// NewAnalyzerWithLogger returns a new analyzer with the given Resizer and Logger.
func NewAnalyzerWithLogger(resizer options.Resizer, logger Logger) Analyzer {
	if logger.Log == nil {
		logger.Log = log.New(ioutil.Discard, "", 0)
	}

	return &smartcropAnalyzer{Resizer: resizer, logger: logger}
}

func (o smartcropAnalyzer) FindBestCrop(img image.Image, width, height int, boosts []BoostRegion) (image.Rectangle, error) {
	if width == 0 && height == 0 {
		return image.Rectangle{}, ErrInvalidDimensions
	}

	// resize image for faster processing
	scale := math.Min(float64(img.Bounds().Dx())/float64(width), float64(img.Bounds().Dy())/float64(height))
	var lowimg *image.RGBA
	var prescalefactor = 1.0

	if prescale {
		// if f := 1.0 / scale / minScale; f < 1.0 {
		// prescalefactor = f
		// }
		if f := prescaleMin / math.Min(float64(img.Bounds().Dx()), float64(img.Bounds().Dy())); f < 1.0 {
			prescalefactor = f
			for idx, boost := range boosts {
				boosts[idx] = BoostRegion{
					X: int(float64(boost.X) * prescalefactor),
					Y: int(float64(boost.Y) * prescalefactor),
					Width:int(float64(boost.Width) * prescalefactor),
					Height: int(float64(boost.Height) * prescalefactor),
					Weight: boost.Weight,
				}
			}

			smallImg := o.Resize(
				img,
				uint(float64(img.Bounds().Dx())*prescalefactor),
				uint(float64(img.Bounds().Dy())*prescalefactor))

			lowimg = ToRGBA(smallImg)
		} else {
			prescalefactor = 1.0
			lowimg = ToRGBA(img)
		}

		o.logger.Log.Println(prescalefactor)

	} else {
		lowimg = ToRGBA(img)
	}

	if o.logger.DebugMode {
		writeImage("png", lowimg, "./smartcrop_prescale.png")
	}

	cropWidth, cropHeight := chop(float64(width)*scale*prescalefactor), chop(float64(height)*scale*prescalefactor)
	realMinScale := math.Min(maxScale, math.Max(1.0/scale, minScale))

	o.logger.Log.Printf("original resolution: %dx%d\n", img.Bounds().Dx(), img.Bounds().Dy())
	o.logger.Log.Printf("scale: %f, cropw: %f, croph: %f, minscale: %f\n", scale, cropWidth, cropHeight, realMinScale)

	topCrop, err := analyse(o.logger, lowimg, cropWidth, cropHeight, realMinScale, boosts)
	if err != nil {
		return topCrop, err
	}

	if prescale == true {
		topCrop.Min.X = int(chop(float64(topCrop.Min.X) / prescalefactor))
		topCrop.Min.Y = int(chop(float64(topCrop.Min.Y) / prescalefactor))
		topCrop.Max.X = int(chop(float64(topCrop.Max.X) / prescalefactor))
		topCrop.Max.Y = int(chop(float64(topCrop.Max.Y) / prescalefactor))
	}

	return topCrop.Canon(), nil
}

func (c Crop) totalScore() float64 {
	return (c.Score.Detail*detailWeight + c.Score.Skin*skinWeight + c.Score.Saturation*saturationWeight + c.Score.Boost * boostWeight) / float64(c.Dx()) / float64(c.Dy())
}

func chop(x float64) float64 {
	if x < 0 {
		return math.Ceil(x)
	}
	return math.Floor(x)
}

func thirds(x float64) float64 {
	x = (math.Mod(x-(1.0/3.0)+1.0, 2.0)*0.5 - 0.5) * 16.0
	return math.Max(1.0-x*x, 0.0)
}

func bounds(l float64) float64 {
	return math.Min(math.Max(l, 0.0), 255)
}

// return ordinary importance and boost area importance
func importance(crop Crop, x, y int) (float64, float64) {
	if crop.Min.X > x || x >= crop.Max.X || crop.Min.Y > y || y >= crop.Max.Y {
		return outsideImportance, outsideImportance
	}

	xf := float64(x-crop.Min.X) / float64(crop.Dx())
	yf := float64(y-crop.Min.Y) / float64(crop.Dy())

	px := math.Abs(0.5-xf) * 2.0
	py := math.Abs(0.5-yf) * 2.0

	dx := math.Max(px-1.0+edgeRadius, 0.0)
	dy := math.Max(py-1.0+edgeRadius, 0.0)
	d := (dx*dx + dy*dy) * edgeWeight

	s := 1.414 - math.Sqrt(px*px+py*py)
	// make a temp copy of s
	sBoost := s
	if ruleOfThirds {
		s += (math.Max(0.0, s+d+0.5) * 1.2) * (thirds(px) + thirds(py))
	}

	return s + d, sBoost * 4
}

func score(sampleOutput *image.RGBA, crop Crop, boosts []BoostRegion) Score {
	width := sampleOutput.Bounds().Dx()
	height := sampleOutput.Bounds().Dy()
	score := Score{}

	downSample := scoreDownSample
	invDownSample := float32(1) / float32(downSample)
	outputHeightDownSample := height * downSample
	outputWidthDownSample := width * downSample

	for y := 0; y < outputHeightDownSample; y += downSample {
		for x := 0; x < outputWidthDownSample; x += downSample {
			sy := int(float32(y) * invDownSample)
			sx := int(float32(x) * invDownSample)

			imp, impBoost := importance(crop, x, y)

			c := sampleOutput.RGBAAt(sx, sy)
			r8 := float64(c.R)
			g8 := float64(c.G)
			b8 := float64(c.B)
			a8 := float64(c.A)

			det := g8 / 255.0

			score.Skin += r8 / 255.0 * (det + skinBias) * imp
			score.Detail += det * imp
			score.Saturation += b8 / 255.0 * (det + saturationBias) * imp
			score.Boost += (a8 / 255) * impBoost
		}
	}

	return score
}

func downSample(input *image.RGBA, factor int) *image.RGBA {
	width := input.Bounds().Dx()/factor
	height := input.Bounds().Dy()/factor
	output := image.NewRGBA(image.Rectangle{Min: image.Point{X:0,Y:0}, Max: image.Point{X:width, Y:height}})
	ifactor2 := 1.0 / (float64(factor) * float64(factor))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			// i := (y * width + x) * 4

			r := uint8(0)
			g := uint8(0)
			b := uint8(0)
			a := uint8(0)

			mr := uint8(0)
			mg := uint8(0)

			for v := 0; v < factor; v++ {
				for u := 0; u < factor; u++ {
					//j := ((y * factor + v) * iwidth + (x * factor + u)) * 4;
					iy := y * factor + v
					ix := x * factor + u
					c := input.RGBAAt(ix, iy)
					r += c.R
					g += c.G
					b += c.B
					a += c.A
					if mr < c.R {
						mr = c.R
					}
					if mg < c.G {
						mg = c.G
					}
					// unused
					//if mb < c.B {
					//	mb = c.B
					//}
				}
			}
			// this is some funky magic to preserve detail a bit more for
			// skin (r) and detail (g). Saturation (b) does not get this boost.
			nc := color.RGBA{
						R:uint8(bounds(float64(r) * ifactor2 * 0.5 + float64(mr) * 0.5)),
						G:uint8(bounds(float64(g) * ifactor2 * 0.7 + float64(mg) * 0.3)),
						B:uint8(bounds(float64(b) * ifactor2)),
						A:uint8(bounds(float64(a) * ifactor2))}

			output.SetRGBA(x, y, nc)
		}
	}

	return output
}


func analyse(logger Logger, img *image.RGBA, cropWidth, cropHeight, realMinScale float64, boosts []BoostRegion) (image.Rectangle, error) {
	o := image.NewRGBA(img.Bounds())

	now := time.Now()
	edgeDetect(img, o)
	logger.Log.Println("Time elapsed edge:", time.Since(now))
	debugOutput(logger.DebugMode, o, "edge")

	now = time.Now()
	skinDetect(img, o)
	logger.Log.Println("Time elapsed skin:", time.Since(now))
	debugOutput(logger.DebugMode, o, "skin")

	now = time.Now()
	saturationDetect(img, o)
	logger.Log.Println("Time elapsed sat:", time.Since(now))
	debugOutput(logger.DebugMode, o, "saturation")

	now = time.Now()
	applyBoosts(boosts, o)
	logger.Log.Println("Time elapsed boost:", time.Since(now))
	debugOutput(logger.DebugMode, o, "boost")


	now = time.Now()
	sampleOutput := downSample(o, scoreDownSample)
	logger.Log.Println("Time elapsed downsample:", time.Since(now))
	debugOutput(logger.DebugMode, sampleOutput, "downSample")

	now = time.Now()
	var topCrop Crop
	topScore := -1.0
	cs := crops(o, cropWidth, cropHeight, realMinScale)
	logger.Log.Println("Time elapsed crops:", time.Since(now), len(cs))

	now = time.Now()
	for _, crop := range cs {
		nowIn := time.Now()
		crop.Score = score(sampleOutput, crop, boosts)
		logger.Log.Println("Time elapsed single-score:", time.Since(nowIn))
		if crop.totalScore() > topScore {
			topCrop = crop
			topScore = crop.totalScore()
		}
	}
	logger.Log.Println("Time elapsed score:", time.Since(now))

	if logger.DebugMode {
		drawDebugCrop(topCrop, o)
		debugOutput(true, o, "final")
	}

	return topCrop.Rectangle, nil
}

func saturation(c color.RGBA) float64 {
	cMax, cMin := uint8(0), uint8(255)
	if c.R > cMax {
		cMax = c.R
	}
	if c.R < cMin {
		cMin = c.R
	}
	if c.G > cMax {
		cMax = c.G
	}
	if c.G < cMin {
		cMin = c.G
	}
	if c.B > cMax {
		cMax = c.B
	}
	if c.B < cMin {
		cMin = c.B
	}

	if cMax == cMin {
		return 0
	}
	maximum := float64(cMax) / 255.0
	minimum := float64(cMin) / 255.0

	l := (maximum + minimum) / 2.0
	d := maximum - minimum

	if l > 0.5 {
		return d / (2.0 - maximum - minimum)
	}

	return d / (maximum + minimum)
}

func cie(c color.RGBA) float64 {
	return 0.5126*float64(c.B) + 0.7152*float64(c.G) + 0.0722*float64(c.R)
}

func skinCol(c color.RGBA) float64 {
	r8, g8, b8 := float64(c.R), float64(c.G), float64(c.B)

	mag := math.Sqrt(r8*r8 + g8*g8 + b8*b8)
	rd := r8/mag - skinColor[0]
	gd := g8/mag - skinColor[1]
	bd := b8/mag - skinColor[2]

	d := math.Sqrt(rd*rd + gd*gd + bd*bd)
	return 1.0 - d
}

func makeCies(img *image.RGBA) []float64 {
	width := img.Bounds().Dx()
	height := img.Bounds().Dy()
	cies := make([]float64, width*height, width*height)
	i := 0
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			cies[i] = cie(img.RGBAAt(x, y))
			i++
		}
	}

	return cies
}

func edgeDetect(i *image.RGBA, o *image.RGBA) {
	width := i.Bounds().Dx()
	height := i.Bounds().Dy()
	cies := makeCies(i)

	var lightness float64
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			if x == 0 || x >= width-1 || y == 0 || y >= height-1 {
				lightness = cie(i.RGBAAt(x, y))
			} else {
				lightness = cies[y*width+x]*4.0 -
					cies[x+(y-1)*width] -
					cies[x-1+y*width] -
					cies[x+1+y*width] -
					cies[x+(y+1)*width]
			}

			nc := color.RGBA{0, uint8(bounds(lightness)), 0, 0}
			o.SetRGBA(x, y, nc)
		}
	}
}

func skinDetect(i *image.RGBA, o *image.RGBA) {
	width := i.Bounds().Dx()
	height := i.Bounds().Dy()

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			lightness := cie(i.RGBAAt(x, y)) / 255.0
			skin := skinCol(i.RGBAAt(x, y))

			c := o.RGBAAt(x, y)
			if skin > skinThreshold && lightness >= skinBrightnessMin && lightness <= skinBrightnessMax {
				r := (skin - skinThreshold) * (255.0 / (1.0 - skinThreshold))
				nc := color.RGBA{uint8(bounds(r)), c.G, c.B, 0}
				o.SetRGBA(x, y, nc)
			} else {
				nc := color.RGBA{0, c.G, c.B, 0}
				o.SetRGBA(x, y, nc)
			}
		}
	}
}

func saturationDetect(i *image.RGBA, o *image.RGBA) {
	width := i.Bounds().Dx()
	height := i.Bounds().Dy()

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			lightness := cie(i.RGBAAt(x, y)) / 255.0
			saturation := saturation(i.RGBAAt(x, y))

			c := o.RGBAAt(x, y)
			if saturation > saturationThreshold && lightness >= saturationBrightnessMin && lightness <= saturationBrightnessMax {
				b := (saturation - saturationThreshold) * (255.0 / (1.0 - saturationThreshold))
				nc := color.RGBA{c.R, c.G, uint8(bounds(b)), 0}
				o.SetRGBA(x, y, nc)
			} else {
				nc := color.RGBA{c.R, c.G, 0, 0}
				o.SetRGBA(x, y, nc)
			}
		}
	}
}

func applyBoosts(boosts []BoostRegion, o *image.RGBA) {
	for _, boost :=  range boosts {
		applyBoost(boost, o)
	}
}

func applyBoost(boost BoostRegion, o *image.RGBA) {
	var x0 = boost.X
	var x1 = boost.X + boost.Width
	var y0 = boost.Y
	var y1 = boost.Y + boost.Height
	var weight = uint8(bounds(boost.Weight * 255.0))

	for y := y0; y < y1; y++ {
		for x := x0; x < x1; x++ {
			c := o.RGBAAt(x, y)
			nc := color.RGBA{c.R, c.G, c.B, weight}
			o.SetRGBA(x, y, nc)
		}
	}
}

func crops(i image.Image, cropWidth, cropHeight, realMinScale float64) []Crop {
	res := []Crop{}
	width := i.Bounds().Dx()
	height := i.Bounds().Dy()

	minDimension := math.Min(float64(width), float64(height))
	var cropW, cropH float64

	if cropWidth != 0.0 {
		cropW = cropWidth
	} else {
		cropW = minDimension
	}
	if cropHeight != 0.0 {
		cropH = cropHeight
	} else {
		cropH = minDimension
	}

	for scale := maxScale; scale >= realMinScale; scale -= scaleStep {
		for y := 0; float64(y)+cropH*scale <= float64(height); y += step {
			for x := 0; float64(x)+cropW*scale <= float64(width); x += step {
				res = append(res, Crop{
					Rectangle: image.Rect(x, y, x+int(cropW*scale), y+int(cropH*scale)),
				})
			}
		}
	}

	return res
}

// toRGBA converts an image.Image to an image.RGBA
func ToRGBA(img image.Image) *image.RGBA {
	switch img.(type) {
	case *image.RGBA:
		return img.(*image.RGBA)
	}
	out := image.NewRGBA(img.Bounds())
	draw.Copy(out, image.Pt(0, 0), img, img.Bounds(), draw.Src, nil)
	return out
}

func CombineImage(images []*image.RGBA) *image.RGBA {
	newY := 0
	totalX := 0
	for _, img := range images {
		rect := img.Bounds()
		if rect.Max.Y > newY {
			newY = rect.Max.Y
		}
		totalX = totalX + rect.Max.X
	}

	newRect := image.Rectangle{
		Min: image.Point{X:0, Y:0},
		Max: image.Point{X:totalX, Y:newY},
	}

	newImage := image.NewRGBA(newRect)
	startX := 0

	for _, img := range images {
		rect := img.Bounds()

		if rect.Max.Y == newY {
			draw.Copy(newImage, image.Pt(startX, 0), img, img.Bounds(), draw.Src, nil)
		} else {
			draw.Copy(newImage, image.Pt(startX, newY - rect.Max.Y), img, img.Bounds(), draw.Src, nil)
		}
		startX = startX + rect.Max.X
	}

	return newImage
}
