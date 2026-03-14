package main

import (
	"fmt"
	"time"
)

func main() {
	now := time.Now()
	rl := NewRateLimiter(10, 1, 100, 5)
	for i := range 10 {
		if !rl.AllowRequest("u1", "o1", now) {
			fmt.Println("oops", 1, i)
		}
	}
	if rl.AllowRequest("u1", "o1", now) {
		fmt.Println("oops", 2)
	}
	now = now.Add(time.Second)
	if !rl.AllowRequest("u1", "o1", now) {
		fmt.Println("oops", 3)
	}
	if rl.AllowRequest("u1", "o1", now) {
		fmt.Println("oops", 4)
	}
	if !rl.AllowRequest("u1", "o2", now) {
		fmt.Println("oops", 5)
	}
	if !rl.AllowRequest("u2", "o1", now) {
		fmt.Println("oops", 6)
	}
	// 93 tokens available in o1
	for i := range 93 {
		if !rl.AllowRequest(UserID(fmt.Sprintf("u%d", i+10)), "o1", now) {
			fmt.Println("oops", 7, i+10)
		}
	}
	if rl.AllowRequest("u99", "o1", now) {
		fmt.Println("oops", 8)
	}
}

const refillRate = time.Second

type TokenBucketOptions struct {
	Cap       float64
	RefillAmt float64
}

func NewTokenBucket(opts TokenBucketOptions) *TokenBucket {
	return &TokenBucket{
		amt:       opts.Cap,
		cap:       opts.Cap,
		refillAmt: opts.RefillAmt,
	}
}

type TokenBucket struct {
	amt          float64
	cap          float64
	lastRefilled time.Time
	refillAmt    float64
}

func (t *TokenBucket) AcquireToken(now time.Time) bool {
	if now.After(t.lastRefilled) {
		timeDelta := now.Sub(t.lastRefilled)
		elapasedSeconds := float64(timeDelta) / float64(refillRate)
		newTokens := t.refillAmt * elapasedSeconds

		t.amt = min(t.amt+newTokens, t.cap)
		t.lastRefilled = now
	}
	if t.amt >= 1.0 {
		t.amt--
		return true
	}
	return false
}

func (t *TokenBucket) ReturnToken() {
	t.amt++
}

type UserRateLimit struct {
	bucket *TokenBucket
}

type UserID string

type OrgRateLimit struct {
	bucket            *TokenBucket
	users             map[UserID]*UserRateLimit
	userBucketOptions TokenBucketOptions
}

func (r *OrgRateLimit) GetUserBucket(user UserID) *UserRateLimit {
	bucket, ok := r.users[user]
	if !ok {
		bucket = new(UserRateLimit)
		bucket.bucket = NewTokenBucket(r.userBucketOptions)
		r.users[user] = bucket
	}
	return bucket
}

type OrgID string

func (r *OrgRateLimit) AcquireToken(userID UserID, now time.Time) bool {
	if !r.bucket.AcquireToken(now) {
		return false
	}
	user := r.GetUserBucket(userID)
	if !user.bucket.AcquireToken(now) {
		r.bucket.ReturnToken()
		return false
	}
	return true
}

type RateLimiter struct {
	orgs              map[OrgID]*OrgRateLimit
	userBucketOptions TokenBucketOptions
	orgBucketOptions  TokenBucketOptions
}

func NewRateLimiter(userTokenCap, userTokenRefillAmt, orgTokenCap, orgTokenRefillAmt float64) *RateLimiter {
	rl := new(RateLimiter)
	rl.orgs = make(map[OrgID]*OrgRateLimit)
	rl.userBucketOptions.Cap = userTokenCap
	rl.userBucketOptions.RefillAmt = userTokenRefillAmt
	rl.orgBucketOptions.Cap = orgTokenCap
	rl.orgBucketOptions.RefillAmt = orgTokenRefillAmt
	return rl
}

func (rl *RateLimiter) getOrg(orgID OrgID) *OrgRateLimit {
	bucket, ok := rl.orgs[orgID]
	if !ok {
		bucket = new(OrgRateLimit)
		bucket.bucket = NewTokenBucket(rl.orgBucketOptions)
		bucket.users = make(map[UserID]*UserRateLimit)
		bucket.userBucketOptions = rl.userBucketOptions
		rl.orgs[orgID] = bucket
	}
	return bucket
}

func (rl *RateLimiter) AllowRequest(userID UserID, orgID OrgID, now time.Time) bool {
	org := rl.getOrg(orgID)
	return org.AcquireToken(userID, now)
}
